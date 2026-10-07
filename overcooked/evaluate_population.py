"""
evaluate_population.py — Evaluate a population of BC clones (extended metrics).

For each clone, across all 5 layouts, runs N episodes and records:

  Original metrics (unchanged):
    - Total reward and deliveries
    - Time to first delivery, mean/min time between deliveries
    - Time to first onion pickup, first onion in pot, first plate pickup
    - Combined action distribution and entropy
    - Training layout generalisation flag

  New metrics — per-agent kitchen event counts:
    Agent 0 = red chef   (agents[0])
    Agent 1 = blue chef  (agents[1])

    For each chef:
      pick_onion          — picked up an onion from a dispenser or counter
      place_onion_counter — put an onion down on a plain counter
      place_onion_pot     — dropped an onion into a pot
      pick_plate          — picked up an empty plate
      place_plate_counter — put an empty plate down on a counter
      fill_plate          — used a plate to scoop ready soup from a pot
      place_soup_counter  — set a plated soup down on a counter (no delivery)
      deliver_soup        — delivered plated soup to the serving area

  New metrics — collisions:
    collisions            — steps where the two agents blocked / contested the
                            same cell (swap, same-target, or body-block)

  New metrics — per-agent step type:
    a{0,1}_idle_steps     — timesteps with action = stay (4)
    a{0,1}_move_ok_steps  — timesteps where agent successfully moved to a new cell
    a{0,1}_interact_steps — timesteps with action = interact (5)

Results are written to:
  evaluation_results.csv   — one row per clone × layout × episode
  evaluation_summary.csv   — one row per clone × layout (episode averages)
  evaluation_summary.json  — same, for programmatic use

Usage:
    python evaluate_population.py \\
        --population pipeline_out/population/clone_population.json \\
        --out-dir    evaluation/ \\
        --n-episodes 10 \\
        --max-steps  400
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

from bc_features import featurize_jaxmarl, FEATURE_DIM, _get_layout_stations
from network import init_model, forward
from checkpoint import load_params_npz


# ── constants ──────────────────────────────────────────────────────────────────

ALL_LAYOUTS = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
]

# JaxMARL observation channels
CH_ONION       = 23   # increases ONLY on dispenser pickup
CH_DISH        = 22   # increases ONLY on dispenser pickup
CH_SOUP        = 21   # ready soup in pot / plated soup held or on counter
CH_POT         = 10   # pot location — STATIC mask (delta always 0, but used as
                      # a spatial filter for at-pot delta computations below)
CH_POT_ONIONS  = 16   # onion count in pot  (+1 for 1st/2nd onion; ≠ 0 at pot cells)
CH_POT_COOKING = 18   # cooking state  (+1 at pot cell when 3rd onion placed)
CH_COUNTER     = 11   # counter occupancy  (d < 0 when item placed; d > 0 when picked up)

# Agent inventory item encodings
# (JaxMARL Overcooked: agent_inv[i] holds one of these values)
INV_EMPTY = 0   # hands empty
INV_ONION = 1   # holding an onion
INV_PLATE = 2   # holding an empty plate
INV_SOUP  = 3   # holding a plated (filled) soup

ACTION_NAMES = ["up", "down", "right", "left", "stay", "interact"]


# ── layout schedule inference ──────────────────────────────────────────────────

def _infer_training_order(clone: Dict) -> Dict[str, Optional[int]]:
    """
    Infer the rotation order in which each layout was first seen during PPO
    fine-tuning, returning a dict mapping layout_name → 0-based position.

    The position is the first interval index at which that layout appeared,
    which equals its position in layout_pool (since rotate cycles in order).

    Three cases:
      rotate  — layout_counts keys give the pool order; position = index in pool.
                All layouts in layout_counts get a position; others get None.
      single  — only the stored "layout" field was used; it gets position 0,
                all others get None.
      random  — order is not recoverable without a training log; all trained
                layouts get position None but trained=True is still indicated
                by returning 0 for all of them (indistinguishable order).
      unknown — fall back gracefully: treat like single if layout field present.

    The schedule type itself is not stored in clone_population.json (a separate
    bug worth fixing), so we infer it from the data:
      - layout_counts has >1 key AND all 5 layouts present → likely rotate/random
      - layout_counts has 1 key OR only one matches layout field → likely single
    """
    layout_counts = clone.get("layout_counts", {})
    last_layout   = clone.get("layout")          # unreliable: last PBT interval

    if not layout_counts:
        # No partition data at all — fall back to last_layout only
        if last_layout:
            return {last_layout: 0}
        return {}

    n_layouts = len(layout_counts)

    if n_layouts == 1:
        # single-layout training
        only = next(iter(layout_counts))
        return {only: 0}

    # Multiple layouts — reconstruct pool order from dict key order.
    # For rotate this is exact; for random we assign None to indicate
    # "trained but order unknown".
    #
    # We distinguish rotate vs random heuristically: if last_layout matches
    # position (total_updates-1) % n_layouts in the key list, it's likely
    # rotate. But since we don't have total_updates here, we can't be sure.
    # We always return positional order; callers can check last_layout if needed.
    pool = list(layout_counts.keys())
    return {layout: idx for idx, layout in enumerate(pool)}

# Kitchen events tracked per-agent (in this order in CSV columns)
KITCHEN_EVENTS = [
    "pick_onion",           # held=="empty", d_onion>0 (dispenser) OR d_counter>0 (counter)
    "place_onion_counter",  # held=="onion",  d_counter<0,  no pot signal
    "place_onion_pot",      # held=="onion",  d_pot_onions_at_pot>0 OR d_pot_cooking_at_pot>0
    "pick_plate",           # held=="empty", d_dish>0 (dispenser) OR d_counter>0 (counter)
    "place_plate_counter",  # held=="plate", d_counter<0,  d_soup_at_pot>=0
    "fill_plate",           # held=="plate", d_soup_at_pot<0  (CH_SOUP drops at pot cell)
    "place_soup_counter",   # held=="soup",  d_counter<0,  no reward
    "deliver_soup",         # reward > 0
]

# Movement direction deltas: action_id → (dx, dy)
MOVE_DELTAS: Dict[int, Tuple[int, int]] = {
    0: ( 0, -1),   # up
    1: ( 0,  1),   # down
    2: ( 1,  0),   # right
    3: (-1,  0),   # left
}
MOVE_ACTIONS    = frozenset(MOVE_DELTAS)   # {0, 1, 2, 3}
STAY_ACTION     = 4
INTERACT_ACTION = 5


# ── per-agent kitchen-event detection ─────────────────────────────────────────

def detect_agent_events(
    action:              int,
    held:                str,    # "empty" | "onion" | "plate" | "soup"
    reward:              float,
    d_onion:             float,  # global Σ CH_ONION delta  — reliable dispenser pickup signal
    d_dish:              float,  # global Σ CH_DISH  delta  — reliable dispenser pickup signal
    d_pot_onions_at_pot: float,  # CH_POT_ONIONS delta restricted to pot cells  (1st/2nd onion)
    d_pot_cooking_at_pot:float,  # CH_POT_COOKING delta restricted to pot cells (3rd onion → > 0)
    d_soup_at_pot:       float,  # CH_SOUP delta restricted to pot cells  (< 0 when soup scooped)
    d_counter:           float,  # global Σ CH_COUNTER delta — < 0 item placed; > 0 item picked up
) -> Tuple[Dict[str, bool], str]:
    """
    State-machine kitchen-event detector.  Returns (events_dict, new_held_state).

    Signal sources — aligned with reward_shaping.py
    ─────────────────────────────────────────────────
    CH_ONION (23) global delta > 0    → dispenser pickup
    CH_DISH  (22) global delta > 0    → dispenser pickup
    CH_POT_ONIONS (16) at-pot delta > 0   → 1st / 2nd onion placed in pot
    CH_POT_COOKING (18) at-pot delta > 0  → 3rd onion placed in pot (cooking start)
    CH_SOUP (21) at-pot delta < 0         → ready soup scooped from pot (fill_plate)
    CH_COUNTER (11) global delta < 0      → any item placed on a counter

    Using at-pot (spatially restricted) deltas for pot events avoids the spurious
    global CH_SOUP increase that occurs while soup is cooking.  This mirrors the
    delta_at_pot() logic in reward_shaping.py exactly.
    """
    ev = {e: False for e in KITCHEN_EVENTS}

    # Delivery: reward signal is the most reliable and is checked first.
    if reward > 0:
        ev["deliver_soup"] = True
        return ev, "empty"

    if action != INTERACT_ACTION:
        return ev, held

    # ── event classification by held state ────────────────────────────────────

    if held == "empty":
        # Dispenser pickups (most common path)
        if d_onion > 0:
            ev["pick_onion"] = True
            return ev, "onion"
        if d_dish > 0:
            ev["pick_plate"] = True
            return ev, "plate"
        # Counter pickup: CH_COUNTER increases when item leaves counter.
        # Without a dedicated "held type" channel we default to onion
        # (the most common counter-staged item).
        if d_counter > 0:
            ev["pick_onion"] = True
            return ev, "onion"

    elif held == "onion":
        # Any at-pot onion channel changing = onion entered pot.
        # d_pot_onions_at_pot covers 1st/2nd; d_pot_cooking_at_pot covers 3rd
        # (mirrors: delta_at_pot(16) > 0 or delta_at_pot(18) > 0 in reward_shaping.py)
        if d_pot_onions_at_pot > 0 or d_pot_cooking_at_pot > 0:
            ev["place_onion_pot"] = True
            return ev, "empty"
        if d_counter < 0:
            ev["place_onion_counter"] = True
            return ev, "empty"
        # Interact failed — keep state.

    elif held == "plate":
        # CH_SOUP decreases at pot cells = agent scooped ready soup into plate.
        # (mirrors: delta_at_pot(21) < 0 in reward_shaping.py)
        if d_soup_at_pot < 0:
            ev["fill_plate"] = True
            return ev, "soup"
        if d_counter < 0:
            ev["place_plate_counter"] = True
            return ev, "empty"
        # Interact with not-yet-ready pot, or failed — keep state.

    elif held == "soup":
        # Delivery already caught above; any counter interaction is a putdown.
        if d_counter < 0:
            ev["place_soup_counter"] = True
            return ev, "empty"

    return ev, held

def detect_events(obs_prev, obs_curr, reward: float) -> Dict:
    """
    Original coarse detection used only for first-event timing milestones.
    Kept unchanged from the original script.
    """
    if obs_prev is None:
        return {"delivery": False, "onion_pickup": False,
                "onion_in_pot": False, "dish_pickup": False}

    def delta(ch):
        if ch >= obs_curr.shape[-1]:
            return 0.0
        return float(jnp.sum(obs_curr[:, :, ch]) - jnp.sum(obs_prev[:, :, ch]))

    return {
        "delivery":     reward > 0,
        "onion_pickup": delta(CH_ONION) > 0,
        "onion_in_pot": delta(CH_POT_ONIONS) != 0,
        "dish_pickup":  delta(CH_DISH)  > 0,
    }


# ── collision detection ────────────────────────────────────────────────────────

def classify_movement(
    pre_state,
    post_state,
    a0_act: int,
    a1_act: int,
) -> Tuple[bool, str, str]:
    """
    Classify each agent's movement outcome in one pass.

    Returns
    ───────
    collision  : bool — a mutual inter-agent block occurred this step
    a0_outcome : str  — one of "move_ok" | "ctr_bump" | "agent_bump" | "non_move"
    a1_outcome : str  — same

    Definitions
    ───────────
    move_ok    — agent chose a move action AND their position changed
    ctr_bump   — agent chose a move action, didn't move, and this was NOT a
                 mutual block (hit a wall or counter)
    agent_bump — agent chose a move action, didn't move, and BOTH agents failed
                 to move on the same step (mutual block / face-off / contest)
    non_move   — agent chose stay or interact (not a movement attempt)

    Why position-based, not target-based
    ──────────────────────────────────────
    The old implementation predicted each agent's intended cell using MOVE_DELTAS
    and compared it to the other agent's position.  If agent_pos is stored as
    (row, col) but the deltas are written for (x, y) = (col, row), every direction
    maps to the wrong neighbour — agent-agent blocks appear as counter bumps.
    This version uses only actual pre/post positions, which are correct regardless
    of coordinate ordering.

    Collision heuristic: both agents tried to move and NEITHER succeeded.
    This covers face-offs, same-cell contests, and symmetric body-blocks.
    It can over-count when both independently hit walls on the same step, but
    that edge case is far less common than genuine agent-agent blocks in the
    constrained Overcooked grid.
    """
    pre_pos  = jnp.array(pre_state.agent_pos)
    post_pos = jnp.array(post_state.agent_pos)

    p0  = tuple(int(x) for x in pre_pos[0])
    p1  = tuple(int(x) for x in pre_pos[1])
    pp0 = tuple(int(x) for x in post_pos[0])
    pp1 = tuple(int(x) for x in post_pos[1])

    a0_tried  = a0_act in MOVE_ACTIONS
    a1_tried  = a1_act in MOVE_ACTIONS
    a0_stuck  = a0_tried and pp0 == p0
    a1_stuck  = a1_tried and pp1 == p1

    # Swap: each agent ended up at the other's starting cell
    swap = (a0_tried and a1_tried
            and pp0 != p0 and pp1 != p1
            and pp0 == p1 and pp1 == p0)

    # Mutual block: both tried to move, neither succeeded, AND they were
    # adjacent (Manhattan distance 1) before the step.
    # Without the adjacency guard, two agents independently hitting walls
    # on the same step would be wrongly counted as a collision — a
    # particular problem in forced_coord where agents are in separate rooms
    # and can never physically meet.
    dist = sum(abs(a - b) for a, b in zip(p0, p1))
    mutual = a0_stuck and a1_stuck and dist == 1

    collision = swap or mutual

    def outcome(tried: bool, stuck: bool) -> str:
        if not tried:
            return "non_move"
        if not stuck:
            return "move_ok"
        return "agent_bump" if collision else "ctr_bump"

    return collision, outcome(a0_tried, a0_stuck), outcome(a1_tried, a1_stuck)





# ── episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    params,
    env,
    layout_name:  str,
    key:          jax.Array,
    max_steps:    int   = 400,
    hidden_sizes: Tuple = (256, 256),
    debug_events: bool  = False,
) -> Dict:
    """
    Run one episode with a clone playing both agent roles (self-play).

    Returns a flat dict of per-episode metrics.  All new per-agent metrics use
    the prefix a0_ (agent_0, red chef) or a1_ (agent_1, blue chef).
    """
    agents = env.agents
    a0, a1 = agents[0], agents[1]

    stations = _get_layout_stations(env)

    key, key_reset = jax.random.split(key)
    obs, state     = env.reset(key_reset)

    grid_w   = env.width
    grid_h   = env.height
    wall_map = np.array(state.wall_map)

    prev_obs0    = None
    prev_obs1    = None
    prev_action0 = STAY_ACTION
    prev_action1 = STAY_ACTION

    # ── original counters ─────────────────────────────────────────────────────
    total_reward  = 0.0
    deliveries    = 0
    action_counts = np.zeros(6, dtype=np.int32)   # combined, kept for entropy

    first_delivery_step = None
    delivery_steps      = []
    first_onion_step    = None
    first_pot_step      = None
    first_dish_step     = None

    # ── new: per-agent kitchen-event counters ─────────────────────────────────
    # ev_counts[agent_idx][event_name] = int count
    ev_counts = {
        0: {e: 0 for e in KITCHEN_EVENTS},
        1: {e: 0 for e in KITCHEN_EVENTS},
    }
    # State machine: what each agent is currently holding.
    # Updated by detect_agent_events each step.
    held_item = {0: "empty", 1: "empty"}

    # ── new: per-agent step-type counters ─────────────────────────────────────
    idle_steps     = {0: 0, 1: 0}
    move_ok_steps  = {0: 0, 1: 0}   # successful moves (position actually changed)
    interact_steps = {0: 0, 1: 0}

    # ── new: collision and bump counters ──────────────────────────────────────
    collisions   = 0              # steps with mutual agent-agent block
    counter_bumps = {0: 0, 1: 0}  # per-agent: stuck due to wall / counter
    agent_bumps   = {0: 0, 1: 0}  # per-agent: stuck due to other agent

    step = 0

    while step < max_steps:
        key, k0, k1, k_step = jax.random.split(key, 4)

        # featurize
        f0 = featurize_jaxmarl(obs[a0], state, 0, stations, wall_map,
                               prev_action=prev_action0,
                               grid_w=grid_w, grid_h=grid_h)
        f1 = featurize_jaxmarl(obs[a1], state, 1, stations, wall_map,
                               prev_action=prev_action1,
                               grid_w=grid_w, grid_h=grid_h)

        logits0, _ = forward(params, f0)
        logits1, _ = forward(params, f1)

        a0_act = int(jax.random.categorical(k0, logits0))
        a1_act = int(jax.random.categorical(k1, logits1))

        action_counts[a0_act] += 1
        action_counts[a1_act] += 1

        actions = {
            a0: jnp.array(a0_act, dtype=jnp.uint32),
            a1: jnp.array(a1_act, dtype=jnp.uint32),
        }

        # snapshot pre-step state (for collision detection)
        pre_state = state

        next_obs, next_state, rewards, dones, _ = env.step(k_step, state, actions)

        step_reward   = float(rewards[a0]) + float(rewards[a1])
        total_reward += step_reward

        # ── backward-compatible timing events (agent 0 observation) ──────────
        coarse = detect_events(prev_obs0, next_obs[a0], float(rewards[a0]))

        if coarse["delivery"]:
            deliveries += 1
            delivery_steps.append(step)
            if first_delivery_step is None:
                first_delivery_step = step

        if coarse["onion_pickup"] and first_onion_step is None:
            first_onion_step = step
        if coarse["onion_in_pot"]  and first_pot_step  is None:
            first_pot_step   = step
        if coarse["dish_pickup"]   and first_dish_step is None:
            first_dish_step  = step

        # ── per-agent kitchen events ──────────────────────────────────────────
        agent_obs_pairs = [
            (0, a0_act, prev_obs0, next_obs[a0], float(rewards[a0])),
            (1, a1_act, prev_obs1, next_obs[a1], float(rewards[a1])),
        ]
        for agent_idx, act, obs_p, obs_c, rw in agent_obs_pairs:
            # Compute channel deltas used by detect_agent_events.
            # All default to 0 on the first step (obs_p is None).
            d_onion = d_dish = d_counter = 0.0
            d_pot_onions_at_pot = d_pot_cooking_at_pot = d_soup_at_pot = 0.0

            if obs_p is not None:
                C = obs_c.shape[-1]
                # Cast to float32 for signed arithmetic.
                # Observations are stored as uint (uint8/uint32), so naive subtraction
                # wraps: e.g. 0 - 1 = 4294967295 instead of -1, which makes every
                # "channel decreased" delta appear as a huge positive number and breaks
                # all conditions that check d_... < 0 (place_onion_counter, fill_plate, etc.)
                obs_c_f = obs_c.astype(jnp.float32)
                obs_p_f = obs_p.astype(jnp.float32)

                def _d(ch):
                    """Global (whole-grid) channel delta."""
                    if ch >= C:
                        return 0.0
                    return float(jnp.sum(obs_c_f[:, :, ch]) - jnp.sum(obs_p_f[:, :, ch]))

                def _d_at_pot(ch):
                    """Channel delta restricted to pot cells (CH_POT spatial mask).
                    Mirrors delta_at_pot() in reward_shaping.py."""
                    if ch >= C or CH_POT >= C:
                        return _d(ch)
                    pot_mask = obs_c_f[:, :, CH_POT] > 0
                    if not jnp.any(pot_mask):
                        return 0.0
                    return float(
                        jnp.sum(obs_c_f[:, :, ch] * pot_mask)
                        - jnp.sum(obs_p_f[:, :, ch] * pot_mask)
                    )

                d_onion             = _d(CH_ONION)
                d_dish              = _d(CH_DISH)
                d_counter           = _d(CH_COUNTER)
                d_pot_onions_at_pot  = _d_at_pot(CH_POT_ONIONS)
                d_pot_cooking_at_pot = _d_at_pot(CH_POT_COOKING)
                d_soup_at_pot        = _d_at_pot(CH_SOUP)

                # ── debug: print on every interact step ───────────────────────
                if debug_events and act == INTERACT_ACTION:
                    C_val   = obs_c.shape[-1]
                    pot_mask_any = bool(jnp.any(obs_c[:, :, CH_POT] > 0)) if CH_POT < C_val else False
                    pot_cells    = int(jnp.sum(obs_c[:, :, CH_POT] > 0))  if CH_POT < C_val else 0
                    print(
                        f"  [DBG] step={step:4d} a{agent_idx} INTERACT  "
                        f"held={held_item[agent_idx]:<6}  reward={rw:.1f}  "
                        f"obs_shape={obs_c.shape}"
                    )
                    print(
                        f"         pot_mask: {pot_cells} cells  "
                        f"CH_POT({CH_POT})_sum={float(jnp.sum(obs_c[:,:,CH_POT])):.1f}"
                    )
                    print(
                        f"         d_onion(ch{CH_ONION})={d_onion:+.3f}  "
                        f"d_dish(ch{CH_DISH})={d_dish:+.3f}  "
                        f"d_counter(ch{CH_COUNTER})={d_counter:+.3f}"
                    )
                    print(
                        f"         d_pot_onions_at_pot(ch{CH_POT_ONIONS})={d_pot_onions_at_pot:+.3f}  "
                        f"d_pot_cooking_at_pot(ch{CH_POT_COOKING})={d_pot_cooking_at_pot:+.3f}  "
                        f"d_soup_at_pot(ch{CH_SOUP})={d_soup_at_pot:+.3f}"
                    )
                    # Also print raw channel sums so we can see if channels have signal at all
                    raw_sums = {
                        f"ch{ch}": float(jnp.sum(obs_c[:, :, ch]))
                        for ch in [CH_ONION, CH_DISH, CH_SOUP, CH_POT,
                                   CH_POT_ONIONS, CH_POT_COOKING, CH_COUNTER]
                        if ch < C_val
                    }
                    print(f"         raw_sums(after): {raw_sums}")

            prev_held = held_item[agent_idx]
            ev, held_item[agent_idx] = detect_agent_events(
                act, held_item[agent_idx], rw,
                d_onion, d_dish,
                d_pot_onions_at_pot, d_pot_cooking_at_pot, d_soup_at_pot,
                d_counter,
            )
            fired = [k for k, v in ev.items() if v]
            if debug_events and act == INTERACT_ACTION:
                print(
                    f"         → fired={fired or 'none'}  "
                    f"held: {prev_held!r} → {held_item[agent_idx]!r}"
                )
            for ev_name, f in ev.items():
                if f:
                    ev_counts[agent_idx][ev_name] += 1

        # ── step-type + collision classification ──────────────────────────────
        col, a0_out, a1_out = classify_movement(pre_state, next_state, a0_act, a1_act)
        if col:
            collisions += 1

        for agent_idx, act, outcome in ((0, a0_act, a0_out), (1, a1_act, a1_out)):
            if outcome == "move_ok":
                move_ok_steps[agent_idx]  += 1
            elif outcome == "ctr_bump":
                counter_bumps[agent_idx]  += 1
            elif outcome == "agent_bump":
                agent_bumps[agent_idx]    += 1
            elif act == STAY_ACTION:
                idle_steps[agent_idx]     += 1
            elif act == INTERACT_ACTION:
                interact_steps[agent_idx] += 1

        # ── advance ───────────────────────────────────────────────────────────
        prev_obs0    = next_obs[a0]
        prev_obs1    = next_obs[a1]
        prev_action0 = a0_act
        prev_action1 = a1_act
        obs          = next_obs
        state        = next_state
        wall_map     = np.array(state.wall_map)
        step        += 1

        if dones.get("__all__", False):
            break

    # ── post-episode summaries ────────────────────────────────────────────────
    inter_delivery_times = [
        delivery_steps[i + 1] - delivery_steps[i]
        for i in range(len(delivery_steps) - 1)
    ] if len(delivery_steps) > 1 else []

    total_actions = int(action_counts.sum())
    probs         = action_counts / max(total_actions, 1)
    entropy       = float(-np.sum(probs * np.log(probs + 1e-8)))

    # ── assemble result dict ──────────────────────────────────────────────────
    result: Dict = {
        # original metrics
        "total_steps":          step,
        "total_reward":         total_reward,
        "deliveries":           deliveries,
        "first_delivery_step":  first_delivery_step,
        "mean_inter_delivery":  float(np.mean(inter_delivery_times))
                                if inter_delivery_times else None,
        "min_inter_delivery":   float(np.min(inter_delivery_times))
                                if inter_delivery_times else None,
        "first_onion_step":     first_onion_step,
        "first_pot_step":       first_pot_step,
        "first_dish_step":      first_dish_step,
        "action_entropy":       entropy,
        "action_counts":        action_counts.tolist(),
        "action_fracs":         (action_counts / max(total_actions, 1)).tolist(),

        # new: collisions (mutual agent-agent blocks) and per-agent bumps
        "collisions":           collisions,
        "a0_counter_bumps":     counter_bumps[0],
        "a1_counter_bumps":     counter_bumps[1],
        "a0_agent_bumps":       agent_bumps[0],
        "a1_agent_bumps":       agent_bumps[1],

        # new: per-agent step types (mutually exclusive and exhaustive)
        "a0_idle_steps":        idle_steps[0],
        "a0_move_ok_steps":     move_ok_steps[0],
        "a0_interact_steps":    interact_steps[0],
        "a1_idle_steps":        idle_steps[1],
        "a1_move_ok_steps":     move_ok_steps[1],
        "a1_interact_steps":    interact_steps[1],
    }

    # new: per-agent kitchen event counts (flat keys: a0_pick_onion, a1_deliver_soup, …)
    for agent_idx in (0, 1):
        for ev_name, count in ev_counts[agent_idx].items():
            result[f"a{agent_idx}_{ev_name}"] = count

    return result


# ── per-clone evaluation ───────────────────────────────────────────────────────

def evaluate_clone(
    clone:        Dict,
    n_episodes:   int   = 10,
    max_steps:    int   = 400,
    hidden_sizes: Tuple = (256, 256),
    seed:         int   = 42,
    verbose:      bool  = True,
    debug_events: bool  = False,
) -> List[Dict]:
    """
    Evaluate one clone across all 5 layouts.
    Returns a list of result dicts (one per layout × episode).

    Each result includes `training_order`: the 0-based position of this layout
    in the clone's PPO rotation schedule, or None if the layout was not trained
    on.  For rotate schedules this reflects the actual interval ordering; for
    random schedules the order is inferred from partition dict key order (which
    is still informative about relative data volume even if not exact timing).
    """
    clone_idx       = clone["clone_idx"]
    checkpoint      = clone["final_checkpoint"]
    last_pbt_layout = clone.get("layout", "unknown")   # last interval only
    training_order  = _infer_training_order(clone)      # layout → 0-based position

    if verbose:
        order_str = ", ".join(
            f"{l}={i}" for l, i in sorted(
                training_order.items(), key=lambda x: x[1]
            )
        )
        print(f"\n[eval] clone {clone_idx:02d}  checkpoint={checkpoint}")
        print(f"       training order: {order_str}")

    like_params = init_model(FEATURE_DIM, 6, hidden_sizes=hidden_sizes)
    params      = load_params_npz(checkpoint, like_params=like_params)

    results = []
    rng     = np.random.default_rng(seed + clone_idx)

    for layout in ALL_LAYOUTS:
        order = training_order.get(layout)   # None if not a training layout

        if verbose:
            if order is None:
                marker = "  ← NOT in training"
            else:
                marker = f"  ← training layout #{order + 1}"
            print(f"  layout: {layout}{marker}")

        try:
            env = make(
                "overcooked",
                layout    = overcooked_layouts[layout],
                max_steps = max_steps + 10,
            )
        except KeyError:
            print(f"  [skip] layout '{layout}' not found in overcooked_layouts")
            continue

        episode_results = []
        for ep in range(n_episodes):
            key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))

            ep_result = run_episode(
                params       = params,
                env          = env,
                layout_name  = layout,
                key          = key,
                max_steps    = max_steps,
                hidden_sizes = hidden_sizes,
                debug_events = debug_events,
            )

            ep_result.update({
                "clone_idx":          clone_idx,
                "layout":             layout,
                "training_order":     order,            # 0-based, or None
                "is_training_layout": order is not None,  # kept for back-compat
                "last_pbt_layout":    last_pbt_layout,  # renamed from bc_layout
                "episode":            ep,
                "checkpoint":         checkpoint,
                "target_score":       clone.get("target_score"),
                "mean_score":         clone.get("mean_score"),
            })
            episode_results.append(ep_result)

            if verbose:
                fd  = ep_result["first_delivery_step"]
                mid = ep_result["mean_inter_delivery"]
                col = ep_result["collisions"]

                # kitchen event shorthand for both agents
                def ev_summary(ag):
                    r = ep_result
                    return (
                        f"onion={r[f'a{ag}_pick_onion']}"
                        f" pot={r[f'a{ag}_place_onion_pot']}"
                        f" plate={r[f'a{ag}_pick_plate']}"
                        f" filled={r[f'a{ag}_fill_plate']}"
                        f" del={r[f'a{ag}_deliver_soup']}"
                    )

                print(
                    f"    ep {ep:02d}  "
                    f"reward={ep_result['total_reward']:5.1f}  "
                    f"deliveries={ep_result['deliveries']}  "
                    f"first_del={fd if fd is not None else 'none':>5}  "
                    f"inter={f'{mid:.1f}' if mid is not None else 'n/a':>6}  "
                    f"collisions={col}  "
                    f"entropy={ep_result['action_entropy']:.3f}"
                )
                print(
                    f"         red(a0)  [{ev_summary(0)}]  "
                    f"idle={ep_result['a0_idle_steps']} "
                    f"move={ep_result['a0_move_ok_steps']} "
                    f"act={ep_result['a0_interact_steps']}"
                )
                print(
                    f"        blue(a1)  [{ev_summary(1)}]  "
                    f"idle={ep_result['a1_idle_steps']} "
                    f"move={ep_result['a1_move_ok_steps']} "
                    f"act={ep_result['a1_interact_steps']}"
                )

        results.extend(episode_results)

    return results


# ── summary statistics ─────────────────────────────────────────────────────────

def summarise(results: List[Dict]) -> List[Dict]:
    """
    Aggregate episode results into per-clone × per-layout summaries.
    All new metrics are included as episode means (and stds where useful).
    """
    from collections import defaultdict
    groups: Dict = defaultdict(list)
    for r in results:
        groups[(r["clone_idx"], r["layout"])].append(r)

    summaries = []
    for (clone_idx, layout), eps in sorted(groups.items()):

        def mean_of(key):
            vals = [e[key] for e in eps if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        def std_of(key):
            vals = [e[key] for e in eps if e.get(key) is not None]
            return float(np.std(vals)) if vals else None

        def pct_none(key):
            return sum(1 for e in eps if e.get(key) is None) / len(eps)

        action_fracs = np.mean(
            [e["action_fracs"] for e in eps], axis=0
        ).tolist()

        summary: Dict = {
            "clone_idx":          clone_idx,
            "layout":             layout,
            "training_order":     eps[0]["training_order"],     # 0-based or None
            "is_training_layout": eps[0]["is_training_layout"], # back-compat bool
            "last_pbt_layout":    eps[0]["last_pbt_layout"],    # renamed from bc_layout
            "target_score":       eps[0]["target_score"],
            "n_episodes":         len(eps),

            # reward
            "mean_reward":        mean_of("total_reward"),
            "std_reward":         std_of("total_reward"),

            # deliveries
            "mean_deliveries":    mean_of("deliveries"),
            "std_deliveries":     std_of("deliveries"),

            # timing: first delivery
            "mean_first_delivery": mean_of("first_delivery_step"),
            "std_first_delivery":  std_of("first_delivery_step"),
            "pct_no_delivery":     pct_none("first_delivery_step"),

            # timing: inter-delivery
            "mean_inter_delivery": mean_of("mean_inter_delivery"),

            # timing: sub-task milestones
            "mean_first_onion":   mean_of("first_onion_step"),
            "mean_first_pot":     mean_of("first_pot_step"),
            "mean_first_dish":    mean_of("first_dish_step"),

            # policy characteristics
            "mean_entropy":       mean_of("action_entropy"),
            "action_fracs":       action_fracs,
            **{f"frac_{ACTION_NAMES[i]}": action_fracs[i] for i in range(6)},

            # new: collisions and per-agent bumps
            "mean_collisions":       mean_of("collisions"),
            "std_collisions":        std_of("collisions"),
            "mean_a0_counter_bumps": mean_of("a0_counter_bumps"),
            "std_a0_counter_bumps":  std_of("a0_counter_bumps"),
            "mean_a1_counter_bumps": mean_of("a1_counter_bumps"),
            "std_a1_counter_bumps":  std_of("a1_counter_bumps"),
            "mean_a0_agent_bumps":   mean_of("a0_agent_bumps"),
            "std_a0_agent_bumps":    std_of("a0_agent_bumps"),
            "mean_a1_agent_bumps":   mean_of("a1_agent_bumps"),
            "std_a1_agent_bumps":    std_of("a1_agent_bumps"),

            # new: per-agent step types
            "mean_a0_idle_steps":     mean_of("a0_idle_steps"),
            "mean_a0_move_ok_steps":  mean_of("a0_move_ok_steps"),
            "mean_a0_interact_steps": mean_of("a0_interact_steps"),
            "mean_a1_idle_steps":     mean_of("a1_idle_steps"),
            "mean_a1_move_ok_steps":  mean_of("a1_move_ok_steps"),
            "mean_a1_interact_steps": mean_of("a1_interact_steps"),
        }

        # new: per-agent kitchen event means (and stds)
        for agent_idx in (0, 1):
            for ev_name in KITCHEN_EVENTS:
                key = f"a{agent_idx}_{ev_name}"
                summary[f"mean_{key}"] = mean_of(key)
                summary[f"std_{key}"]  = std_of(key)

        summaries.append(summary)

    return summaries


# ── CSV / JSON writers ─────────────────────────────────────────────────────────

def write_episode_csv(results: List[Dict], path: str):
    """Write one row per episode.  List fields (action_counts, action_fracs)
    are flattened into per-action columns.  All other fields are written as-is."""
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    flat = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("action_counts", "action_fracs")}
        for i, name in enumerate(ACTION_NAMES):
            row[f"count_{name}"] = r["action_counts"][i]
            row[f"frac_{name}"]  = r["action_fracs"][i]
        flat.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    print(f"[eval] episode results  → {path}")


def write_summary_csv(summaries: List[Dict], path: str):
    if not summaries:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = [{k: v for k, v in s.items() if k != "action_fracs"}
            for s in summaries]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    print(f"[eval] summary          → {path}")


def write_summary_json(summaries: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[eval] summary JSON     → {path}")


# ── pretty-print helpers ───────────────────────────────────────────────────────

# Short abbreviations for the kitchen-events table header
_EV_ABBREV: Dict[str, str] = {
    "pick_onion":           "PO",
    "place_onion_counter":  "OC",
    "place_onion_pot":      "OP",
    "pick_plate":           "PP",
    "place_plate_counter":  "PC",
    "fill_plate":           "FP",
    "place_soup_counter":   "SC",
    "deliver_soup":         "DS",
}

def _f(v, spec=".1f") -> str:
    """Format a nullable float; return 'n/a' if None."""
    return f"{v:{spec}}" if v is not None else "n/a"

def _fpm(mean, std, spec=".1f") -> str:
    """Format mean ± std; return 'n/a' if mean is None."""
    if mean is None:
        return "n/a"
    s = f"{mean:{spec}}"
    if std is not None:
        s += f" ±{std:{spec}}"
    return s

def _rule(width: int, char: str = "─") -> str:
    return char * width

def _divider(width: int) -> str:
    return "·" * width


def print_performance_table(summaries: List[Dict]) -> None:
    """
    Table 1 — Performance Overview
    One row per clone × layout, showing reward, deliveries, collisions,
    first-delivery step, inter-delivery gap, action entropy, and training
    order (1st, 2nd, … or 'no' if not a training layout).
    """
    # column widths
    CW = 5    # clone
    LW = 22   # layout
    RW = 15   # reward ±std
    DW = 12   # deliveries ±std
    KW = 13   # collisions ±std
    FW = 9    # first delivery
    IW = 9    # inter delivery
    EW = 8    # entropy
    TW = 6    # train order

    total = CW + LW + RW + DW + KW + FW + IW + EW + TW + 16  # 16 = 8 separators × 2

    header = (
        f"  {'clone':>{CW}}  {'layout':<{LW}}  "
        f"{'reward (±std)':>{RW}}  "
        f"{'deliveries':>{DW}}  "
        f"{'collisions':>{KW}}  "
        f"{'first_del':>{FW}}  "
        f"{'inter_del':>{IW}}  "
        f"{'entropy':>{EW}}  "
        f"{'order':>{TW}}"
    )

    print(f"\n  PERFORMANCE OVERVIEW  (order = position in PPO layout rotation, 1-based)")
    print(f"  {_rule(total)}")
    print(header)
    print(f"  {_rule(total)}")

    prev_clone = None
    for s in summaries:
        if prev_clone is not None and s["clone_idx"] != prev_clone:
            print(f"  {_divider(total)}")
        prev_clone = s["clone_idx"]

        order = s.get("training_order")
        if order is None:
            order_str = "no"
        else:
            order_str = f"#{order + 1}"   # 1-based for display

        print(
            f"  {s['clone_idx']:>{CW}}  {s['layout']:<{LW}}  "
            f"{_fpm(s['mean_reward'],    s['std_reward']):>{RW}}  "
            f"{_fpm(s['mean_deliveries'],s['std_deliveries']):>{DW}}  "
            f"{_fpm(s['mean_collisions'],s['std_collisions']):>{KW}}  "
            f"{_f(s['mean_first_delivery']):>{FW}}  "
            f"{_f(s['mean_inter_delivery']):>{IW}}  "
            f"{_f(s['mean_entropy'], '.3f'):>{EW}}  "
            f"{order_str:>{TW}}"
        )

    print(f"  {_rule(total)}")


def print_steps_table(summaries: List[Dict]) -> None:
    """
    Table 2 — Time Allocation
    Two sub-rows per clone × layout (one per agent).  All five step categories
    are mutually exclusive and sum to the episode length.
    """
    CW = 5    # clone
    LW = 22   # layout
    AW = 5    # agent label
    IW = 7    # idle
    MW = 8    # move_ok
    TW = 9    # interact
    BW = 8    # ctr_bump
    GW = 9    # agent_bump

    total = CW + LW + AW + IW + MW + TW + BW + GW + 14

    header = (
        f"  {'clone':>{CW}}  {'layout':<{LW}}  "
        f"{'agent':<{AW}}  "
        f"{'idle':>{IW}}  "
        f"{'move_ok':>{MW}}  "
        f"{'interact':>{TW}}  "
        f"{'ctr_bump':>{BW}}  "
        f"{'agt_bump':>{GW}}"
    )

    print(f"\n  TIME ALLOCATION  (mean steps / episode · agent_0=red · agent_1=blue)")
    print(f"  move_ok=successful move · ctr_bump=hit wall/counter · agt_bump=blocked by other agent")
    print(f"  {_rule(total)}")
    print(header)
    print(f"  {_rule(total)}")

    prev_clone = None
    for s in summaries:
        if prev_clone is not None and s["clone_idx"] != prev_clone:
            print(f"  {_divider(total)}")
        prev_clone = s["clone_idx"]

        for ag, label in ((0, "red"), (1, "blue")):
            clone_col  = f"{s['clone_idx']:>{CW}}" if ag == 0 else " " * CW
            layout_col = f"{s['layout']:<{LW}}" if ag == 0 else " " * LW

            print(
                f"  {clone_col}  {layout_col}  "
                f"{label:<{AW}}  "
                f"{_f(s.get(f'mean_a{ag}_idle_steps')):>{IW}}  "
                f"{_f(s.get(f'mean_a{ag}_move_ok_steps')):>{MW}}  "
                f"{_f(s.get(f'mean_a{ag}_interact_steps')):>{TW}}  "
                f"{_f(s.get(f'mean_a{ag}_counter_bumps')):>{BW}}  "
                f"{_f(s.get(f'mean_a{ag}_agent_bumps')):>{GW}}"
            )

    print(f"  {_rule(total)}")


def print_events_table(summaries: List[Dict]) -> None:
    """
    Table 3 — Kitchen Events
    Two sub-rows per clone × layout (one per agent) showing mean counts of each
    kitchen event.  Column headers use two-letter abbreviations; a legend is
    printed above the table.
    """
    abbrevs = [_EV_ABBREV[e] for e in KITCHEN_EVENTS]
    EVC = 5   # width per event column
    CW  = 5   # clone
    LW  = 22  # layout
    AW  = 5   # agent

    # total = fixed columns + event columns + separators
    ev_block = len(abbrevs) * (EVC + 2) - 2
    total    = CW + LW + AW + ev_block + 6  # 3 fixed separators × 2

    # legend
    legend_parts = [f"{v}={k}" for k, v in _EV_ABBREV.items()]
    # wrap legend into two lines of ≤ 90 chars each
    legend_lines: List[str] = []
    line = "  Legend:  "
    for part in legend_parts:
        candidate = line + part + "   "
        if len(candidate) > 92 and line.strip() != "Legend:":
            legend_lines.append(line.rstrip())
            line = "           " + part + "   "
        else:
            line = candidate
    legend_lines.append(line.rstrip())

    header = (
        f"  {'clone':>{CW}}  {'layout':<{LW}}  {'agent':<{AW}}  "
        + "  ".join(f"{a:>{EVC}}" for a in abbrevs)
    )

    print(f"\n  KITCHEN EVENTS  (mean count / episode · agent_0=red · agent_1=blue)")
    for ll in legend_lines:
        print(ll)
    print(f"  {_rule(total)}")
    print(header)
    print(f"  {_rule(total)}")

    prev_clone = None
    for s in summaries:
        if prev_clone is not None and s["clone_idx"] != prev_clone:
            print(f"  {_divider(total)}")
        prev_clone = s["clone_idx"]

        for ag, label in ((0, "red"), (1, "blue")):
            clone_col  = f"{s['clone_idx']:>{CW}}" if ag == 0 else " " * CW
            layout_col = f"{s['layout']:<{LW}}" if ag == 0 else " " * LW
            ev_cols    = "  ".join(
                f"{_f(s.get(f'mean_a{ag}_{e}')):>{EVC}}"
                for e in KITCHEN_EVENTS
            )
            print(
                f"  {clone_col}  {layout_col}  {label:<{AW}}  {ev_cols}"
            )

    print(f"  {_rule(total)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate a population of BC clones across all layouts "
                    "(extended: per-agent events, collisions, idle/moving steps)."
    )
    ap.add_argument("--population",  type=str, required=True,
                    help="Path to clone_population.json")
    ap.add_argument("--out-dir",     type=str, default="evaluation/",
                    help="Output directory for CSV and JSON results")
    ap.add_argument("--n-episodes",  type=int, default=10,
                    help="Episodes per clone per layout")
    ap.add_argument("--max-steps",   type=int, default=400,
                    help="Max steps per episode")
    ap.add_argument("--hidden",      type=int, nargs="+", default=[256, 256],
                    help="Network hidden layer sizes")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--clone-idx",   type=int, default=None,
                    help="Evaluate only this clone index (default: all)")
    ap.add_argument("--layout",      type=str, default=None,
                    help="Evaluate only this layout (default: all)")
    ap.add_argument("--quiet",        action="store_true")
    ap.add_argument("--debug-events", action="store_true",
                    help="Print detailed channel deltas on every interact step "
                         "(very verbose; use with --n-episodes 1 --clone-idx N)")
    return ap.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.population) as f:
        population = json.load(f)

    if args.clone_idx is not None:
        population = [c for c in population
                      if c["clone_idx"] == args.clone_idx]
        if not population:
            print(f"[eval] clone {args.clone_idx} not found")
            return

    hidden_sizes = tuple(args.hidden)
    all_results: List[Dict] = []
    start = time.time()

    print(f"[eval] {len(population)} clones  ×  "
          f"{len(ALL_LAYOUTS)} layouts  ×  "
          f"{args.n_episodes} episodes")
    print(f"[eval] new metrics: per-agent kitchen events, collisions, "
          f"idle/moving/interact steps")
    print(f"[eval] agent_0 = red chef  |  agent_1 = blue chef")

    for clone in population:
        results = evaluate_clone(
            clone        = clone,
            n_episodes   = args.n_episodes,
            max_steps    = args.max_steps,
            hidden_sizes = hidden_sizes,
            seed         = args.seed,
            verbose      = not args.quiet and not args.debug_events,
            debug_events = args.debug_events,
        )
        if args.layout:
            results = [r for r in results if r["layout"] == args.layout]
        all_results.extend(results)

    summaries = summarise(all_results)

    os.makedirs(args.out_dir, exist_ok=True)
    write_episode_csv(
        all_results, os.path.join(args.out_dir, "evaluation_results.csv"))
    write_summary_csv(
        summaries,   os.path.join(args.out_dir, "evaluation_summary.csv"))
    write_summary_json(
        summaries,   os.path.join(args.out_dir, "evaluation_summary.json"))

    print_performance_table(summaries)
    print_steps_table(summaries)
    print_events_table(summaries)

    elapsed = time.time() - start
    print(f"\n[eval] done in {elapsed:.1f}s  —  results in {args.out_dir}")


if __name__ == "__main__":
    main()