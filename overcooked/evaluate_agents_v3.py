"""Evaluate Baseline, CF, CCM, and CCM+CF against held-out partner clones.

Belief remains fixed within each episode and can be swept independently of
true partner competence. CCM and CCM+CF select low/middle/high specialists.
Use the commands in overcooked/README.md; --help lists evaluation options.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

from bc_features import FEATURE_DIM, _get_layout_stations, featurize_jaxmarl
from checkpoint import load_params_npz
from network import init_model, forward
from evaluate_population import (
    classify_movement,
    KITCHEN_EVENTS,
    STAY_ACTION,
    INTERACT_ACTION,
    MOVE_ACTIONS,
    MOVE_DELTAS,
)

ACT_DIM = 6

# ── layout name map ───────────────────────────────────────────────────────────
# training script layout field → JaxMARL internal name
LAYOUT_NAME_MAP = {
    "Cramped_Room":        "cramped_room",
    "cramped_room":        "cramped_room",
    "Coord_Ring":          "coord_ring",
    "coord_ring":          "coord_ring",
    "coordination_ring":   "coord_ring",
    "Asymm_Advantages":    "asymm_advantages",
    "asymm_advantages":    "asymm_advantages",
    "asymmetric_advantages": "asymm_advantages",
    "Counter_Circuit":     "counter_circuit",
    "counter_circuit":     "counter_circuit",
    "Forced_Coordination": "forced_coord",
    "forced_coord":        "forced_coord",
    "forced_coordination": "forced_coord",
}

# training sets JSON layout key → internal name
TRAIN_SET_LAYOUT_MAP = {
    "Cramped_Room":        "cramped_room",
    "Coord_Ring":          "coord_ring",
    "Asymm_Advantages":    "asymm_advantages",
    "Counter_Circuit":     "counter_circuit",
    "Forced_Coordination": "forced_coord",
}

# internal layout name → training sets JSON key
LAYOUT_TO_TRAIN_KEY = {v: k for k, v in TRAIN_SET_LAYOUT_MAP.items()}


# ── threshold computation ─────────────────────────────────────────────────────

def compute_thresholds(
    train_sets:  Dict,
    layout_key:  str,
    metric:      str,
    clone_scores: Dict,
) -> Tuple[float, float]:
    """
    Recompute Q1 and Q2 thresholds from the training set using the same logic
    as 01_setup_partners_and_splits.ipynb.

    Uses int(Q1) == int(Q2) tie-breaking, matching the notebook exactly.
    """
    train_clones = train_sets[layout_key]["Train"]["All"]
    values = []
    for clone_entry in train_clones:
        # entries are either "clone.npz" strings or ["clone.npz", "role"] pairs
        if isinstance(clone_entry, list):
            clone_file, _ = clone_entry
        else:
            clone_file = clone_entry
        clone_key = clone_file.replace(".npz", "")
        entry = clone_scores.get(clone_key, {})
        for role in ("clone_0", "clone_1"):
            score = entry.get(role, {}).get(metric)
            if score is not None:
                values.append(float(score))

    if not values:
        raise ValueError(
            f"No training scores found for layout={layout_key} metric={metric}"
        )

    values_arr = np.array(values)
    Q1 = float(np.quantile(values_arr, 0.33))
    Q2 = float(np.quantile(values_arr, 0.66))

    # tie-breaking: matches notebook's int(Q1) == int(Q2) condition
    if int(Q1) == int(Q2):
        temp = [v for v in values if v > Q1]
        if temp:
            Q2 = float(np.quantile(temp, 0.5))

    return Q1, Q2


def score_to_group(score: float, Q1: float, Q2: float) -> str:
    if score <= Q1:
        return "Low"
    elif score <= Q2:
        return "Middle"
    else:
        return "High"


# ── model loading ─────────────────────────────────────────────────────────────

def _model_path(models_dir: str, output_name: str) -> str:
    return os.path.join(models_dir, output_name, "population", "final.npz")


def _result_json_path(models_dir: str, output_name: str) -> str:
    return os.path.join(models_dir, output_name, "population",
                         "agent_population.json")


def load_extra_feature_stats(
    models_dir:  str,
    output_name: str,
) -> Dict[str, Tuple[float, float]]:
    """
    Load the {feature_name: (mean, std)} normalization stats that
    agent_population.py computed from this agent's TRAINING pool and
    saved alongside its checkpoint. Returns {} if the agent has no extra
    features, or if the result file predates this field (in which case
    raw, unnormalized scores are used — same as before this change).
    """
    path = _result_json_path(models_dir, output_name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        result = json.load(f)
    raw_stats = result.get("extra_feature_stats", {})
    return {feat: tuple(stat) for feat, stat in raw_stats.items()}


def _zscore(value: float, feature: str,
            stats: Dict[str, Tuple[float, float]]) -> float:
    if feature not in stats:
        return value
    mean, std = stats[feature]
    std = std if std > 1e-8 else 1.0
    return (value - mean) / std


def load_agent_params(
    path:         str,
    obs_dim:      int,
    hidden_sizes: Tuple[int, ...],
    seed:         int = 0,
) -> object:
    like   = init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes,
                        key=jax.random.PRNGKey(seed))
    return load_params_npz(path, like_params=like)


def detect_obs_dim(npz_path: str) -> int:
    data = np.load(npz_path, allow_pickle=True)
    return int(data["layers/0/W"].shape[1])


# ── action selection ──────────────────────────────────────────────────────────

def select_action(
    params, obs_arr, state, player_idx,
    stations, wall_map, grid_w, grid_h,
    cook_time, extra_values, key,
):
    feat = featurize_jaxmarl(
        obs=obs_arr, state=state, player_idx=player_idx,
        stations=stations, wall_map=wall_map,
        grid_w=grid_w, grid_h=grid_h, cook_time=cook_time,
    )
    if extra_values:
        feat = np.concatenate([feat, np.array(extra_values, dtype=np.float32)])
    logits, _ = forward(params, jnp.asarray(feat))
    key, k = jax.random.split(key)
    return int(jax.random.categorical(k, logits)), key


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(
    agent_params,
    clone_params,
    agent_extra:   List[float],
    clone_extra:   List[float],
    agent_seat:    int,           # 0 or 1
    env,
    stations:      dict,
    wall_map:      np.ndarray,
    cook_time:     int,
    max_steps:     int,
    seed:          int,
) -> Dict:
    """
    Run one episode. agent_seat determines which seat the adaptive agent
    occupies; clone occupies the other seat.

    Returns a flat dict of all per-episode metrics including:
      - kitchen event counts (agent_ and clone_ prefixed)
      - inter-action gaps (mean/std per event type)
      - step types, collisions
      - action transitions (raw counts dict, saved separately)
    """
    clone_seat = 1 - agent_seat
    agent_key  = f"agent_{agent_seat}"
    clone_key  = f"agent_{clone_seat}"

    key = jax.random.PRNGKey(seed)
    key, k_r = jax.random.split(key)
    obs, state = env.reset(k_r)

    # ── counters ──────────────────────────────────────────────────────────────
    ev_counts = {"agent": {e: 0 for e in KITCHEN_EVENTS},
                 "clone": {e: 0 for e in KITCHEN_EVENTS}}

    idle_steps   = {"agent": 0, "clone": 0}
    move_ok      = {"agent": 0, "clone": 0}
    interact_s   = {"agent": 0, "clone": 0}
    ctr_bumps    = {"agent": 0, "clone": 0}
    agt_bumps    = {"agent": 0, "clone": 0}
    collisions   = 0
    both_idle    = 0
    total_reward = 0

    # timing
    first_delivery = None
    delivery_steps = []

    # inter-action gaps: list of gaps per event type
    agent_gaps:  Dict[str, List[int]] = {e: [] for e in KITCHEN_EVENTS}
    last_action_step = {"agent": None}  # step of most recent productive action

    # action transitions: transitions[src][dst] = count
    agent_transitions: Dict[str, Dict[str, int]] = {
        e: {e2: 0 for e2 in KITCHEN_EVENTS} for e in KITCHEN_EVENTS
    }
    last_event = {"agent": None}  # most recent productive event

    # JaxMARL OBJECT_TO_INDEX values → held string
    _INV_TO_HELD = {1: "empty", 3: "onion", 5: "plate", 9: "soup"}

    def _held_from_state(st, seat: int) -> str:
        return _INV_TO_HELD.get(int(st.agent_inv[seat]), "empty")

    for step in range(max_steps):
        key, k_ag, k_cl, k_step = jax.random.split(key, 4)

        pre_state = state

        # read held state directly from ground-truth inventory before the step
        pre_held_agent = _held_from_state(pre_state, agent_seat)
        pre_held_clone = _held_from_state(pre_state, clone_seat)

        # agent action
        a_act, _ = select_action(
            agent_params, np.array(obs[agent_key]), state, agent_seat,
            stations, wall_map, env.width, env.height,
            cook_time, agent_extra, k_ag,
        )
        # clone action
        c_act, _ = select_action(
            clone_params, np.array(obs[clone_key]), state, clone_seat,
            stations, wall_map, env.width, env.height,
            cook_time, clone_extra, k_cl,
        )

        actions = {
            agent_key: jnp.array(a_act, dtype=jnp.uint32),
            clone_key: jnp.array(c_act, dtype=jnp.uint32),
        }
        key, k_step = jax.random.split(key)
        obs, state, rewards, dones, _ = env.step(k_step, state, actions)
        total_reward += float(rewards[agent_key]) + float(rewards[clone_key])

        # ── kitchen events ────────────────────────────────────────────────────
        # Read post-step inventory directly from state for both agents
        post_held_agent = _held_from_state(state, agent_seat)
        post_held_clone = _held_from_state(state, clone_seat)

        pot_positions     = stations.get("pots", set())
        serving_positions = stations.get("serving_counters", set())

        for who, seat, act, o_key, rw, pre_held, post_held in [
            ("agent", agent_seat, a_act, agent_key,
             float(rewards[agent_key]), pre_held_agent, post_held_agent),
            ("clone", clone_seat, c_act, clone_key,
             float(rewards[clone_key]), pre_held_clone, post_held_clone),
        ]:
            a_pos = (int(pre_state.agent_pos[seat][0]),
                     int(pre_state.agent_pos[seat][1]))
            a_dir = (int(pre_state.agent_dir[seat][0]),
                     int(pre_state.agent_dir[seat][1]))

            evs = detect_events_from_inv(
                act               = act,
                pre_inv           = pre_held,
                post_inv          = post_held,
                reward            = rw,
                agent_pos         = a_pos,
                agent_dir         = a_dir,
                pot_positions     = pot_positions,
                serving_positions = serving_positions,
            )

            for ev_name, fired in evs.items():
                if fired:
                    ev_counts[who][ev_name] += 1

                    if who == "agent":
                        if last_action_step["agent"] is not None:
                            agent_gaps[ev_name].append(
                                step - last_action_step["agent"]
                            )
                        last_action_step["agent"] = step

                        if last_event["agent"] is not None:
                            agent_transitions[last_event["agent"]][ev_name] += 1
                        last_event["agent"] = ev_name

        # delivery timing
        if rewards[agent_key] > 0 or rewards[clone_key] > 0:
            delivery_steps.append(step)
            if first_delivery is None:
                first_delivery = step

        # ── movement / step types ─────────────────────────────────────────────
        col, a0_out, a1_out = classify_movement(
            pre_state, state,
            a_act if agent_seat == 0 else c_act,
            c_act if agent_seat == 0 else a_act,
        )
        if col:
            collisions += 1

        seat_outcomes = {
            agent_seat: a0_out if agent_seat == 0 else a1_out,
            clone_seat: a1_out if agent_seat == 0 else a0_out,
        }

        # pre-step positions for directional bump detection
        pre_agent_pos = (int(pre_state.agent_pos[agent_seat][0]),
                         int(pre_state.agent_pos[agent_seat][1]))
        pre_clone_pos = (int(pre_state.agent_pos[clone_seat][0]),
                         int(pre_state.agent_pos[clone_seat][1]))

        for who, seat, act in [("agent", agent_seat, a_act),
                                ("clone", clone_seat, c_act)]:
            out = seat_outcomes[seat]
            if out == "move_ok":
                move_ok[who]   += 1
            elif out == "ctr_bump":
                ctr_bumps[who] += 1
            elif out == "agent_bump":
                # directional: did THIS agent try to walk into the OTHER's cell?
                if act in MOVE_ACTIONS:
                    dx, dy   = MOVE_DELTAS[act]
                    my_pos   = pre_agent_pos if who == "agent" else pre_clone_pos
                    other_pos = pre_clone_pos if who == "agent" else pre_agent_pos
                    intended = (my_pos[0] + dx, my_pos[1] + dy)
                    if intended == other_pos:
                        agt_bumps[who] += 1
            elif act == STAY_ACTION:
                idle_steps[who] += 1
            elif act == INTERACT_ACTION:
                interact_s[who] += 1

        # neither agent idle: both took a non-stay action
        if a_act != STAY_ACTION and c_act != STAY_ACTION:
            both_idle += 1

        if dones.get("__all__", False):
            break

    # ── inter-delivery gaps ───────────────────────────────────────────────────
    inter_del = [delivery_steps[i+1] - delivery_steps[i]
                 for i in range(len(delivery_steps) - 1)]

    # ── assemble result ───────────────────────────────────────────────────────
    result = {
        "total_reward":       total_reward,
        "total_steps":        step + 1,
        "deliveries":         ev_counts["agent"]["deliver_soup"]
                              + ev_counts["clone"]["deliver_soup"],
        "first_delivery":     first_delivery,
        "mean_inter_delivery": float(np.mean(inter_del)) if inter_del else None,
        "collisions":          collisions,
        "neither_idle_steps":  both_idle,
    }

    # per-agent event counts (agent_ and clone_ prefixed)
    for who in ("agent", "clone"):
        for ev in KITCHEN_EVENTS:
            result[f"{who}_{ev}"] = ev_counts[who][ev]

    # per-agent step types
    for who in ("agent", "clone"):
        result[f"{who}_idle"]     = idle_steps[who]
        result[f"{who}_move_ok"]  = move_ok[who]
        result[f"{who}_interact"] = interact_s[who]
        result[f"{who}_ctr_bump"] = ctr_bumps[who]
        result[f"{who}_agt_bump"] = agt_bumps[who]

    # difference metrics (agent minus clone) for each productive action
    for ev in KITCHEN_EVENTS:
        result[f"diff_{ev}"] = ev_counts["agent"][ev] - ev_counts["clone"][ev]

    # inter-action gap stats per event
    for ev in KITCHEN_EVENTS:
        gaps = agent_gaps[ev]
        result[f"gap_mean_{ev}"] = float(np.mean(gaps)) if gaps else None
        result[f"gap_std_{ev}"]  = float(np.std(gaps))  if gaps else None

    # store transitions separately (returned as nested dict)
    result["_transitions"] = agent_transitions

    return result


# ── CSV helpers ───────────────────────────────────────────────────────────────

RESULT_COLS = (
    # trial identity
    ["script", "agent_name", "agent_type", "layout",
     "clone_name", "clone_role", "clone_score", "clone_group",
     "agent_score_input", "agent_group", "trial", "agent_seat", "aligned",
     # performance
     "total_reward", "total_steps", "deliveries",
     "first_delivery", "mean_inter_delivery", "collisions", "neither_idle_steps",
    ]
    # agent kitchen events
    + [f"agent_{e}" for e in KITCHEN_EVENTS]
    # clone kitchen events
    + [f"clone_{e}" for e in KITCHEN_EVENTS]
    # difference
    + [f"diff_{e}" for e in KITCHEN_EVENTS]
    # agent step types
    + ["agent_idle", "agent_move_ok", "agent_interact",
       "agent_ctr_bump", "agent_agt_bump"]
    # clone step types
    + ["clone_idle", "clone_move_ok", "clone_interact",
       "clone_ctr_bump", "clone_agt_bump"]
    # inter-action gaps
    + [f"gap_mean_{e}" for e in KITCHEN_EVENTS]
    + [f"gap_std_{e}"  for e in KITCHEN_EVENTS]
)

TRANSITION_COLS = [
    "script", "agent_name", "clone_name", "clone_role",
    "agent_score_input", "agent_group", "trial", "agent_seat", "aligned",
    "from_event", "to_event", "count",
]


def detect_events_from_inv(
    act:             int,
    pre_inv:         str,    # "empty" | "onion" | "plate" | "soup"
    post_inv:        str,
    reward:          float,
    agent_pos:       Tuple[int, int],   # (x, y) pre-step position
    agent_dir:       Tuple[int, int],   # (dx, dy) facing direction
    pot_positions:   set,               # set of (x, y) pot cells
    serving_positions: set,             # set of (x, y) serving counter cells
) -> Dict[str, bool]:
    """
    Per-agent kitchen event detector using ground-truth inventory transitions
    and agent position/direction to disambiguate placement targets.

    Each event is determined by comparing pre_inv and post_inv (read directly
    from state.agent_inv before and after the step). For placement events
    (onion→empty, plate→empty, soup→empty) we use the agent's facing direction
    to determine which cell they interacted with, resolving pot vs counter
    without relying on global observation channel deltas.

    This is fully per-agent and immune to the global-channel-delta problem.
    """
    ev = {e: False for e in KITCHEN_EVENTS}

    if act != INTERACT_ACTION:
        return ev

    # cell the agent is facing / interacting with
    target_cell = (agent_pos[0] + agent_dir[0], agent_pos[1] + agent_dir[1])

    # ── delivery ──────────────────────────────────────────────────────────────
    # Must be facing a serving counter AND holding soup AND reward fired.
    # The spatial check makes this per-agent even if both receive the reward.
    if (reward > 0
            and pre_inv == "soup"
            and post_inv == "empty"
            and target_cell in serving_positions):
        ev["deliver_soup"] = True
        return ev

    # ── pickups (empty → something) ───────────────────────────────────────────
    if pre_inv == "empty" and post_inv == "onion":
        ev["pick_onion"] = True
        return ev

    if pre_inv == "empty" and post_inv == "plate":
        ev["pick_plate"] = True
        return ev

    # ── onion placements (onion → empty) ─────────────────────────────────────
    if pre_inv == "onion" and post_inv == "empty":
        if target_cell in pot_positions:
            ev["place_onion_pot"] = True
        else:
            ev["place_onion_counter"] = True
        return ev

    # ── fill plate (plate → soup) ─────────────────────────────────────────────
    if pre_inv == "plate" and post_inv == "soup":
        ev["fill_plate"] = True
        return ev

    # ── plate placed on counter (plate → empty) ───────────────────────────────
    if pre_inv == "plate" and post_inv == "empty":
        ev["place_plate_counter"] = True
        return ev

    # ── soup placed on counter (soup → empty, no reward) ─────────────────────
    if pre_inv == "soup" and post_inv == "empty" and reward == 0:
        ev["place_soup_counter"] = True
        return ev

    return ev


def _open_csv(path: str, cols: List[str], overwrite: bool) -> Tuple[csv.DictWriter, object]:
    """Open a CSV for append or overwrite. Returns (writer, file_handle)."""
    exists = os.path.exists(path)
    mode   = "w" if overwrite or not exists else "a"
    fh     = open(path, mode, newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if mode == "w" or not exists:
        writer.writeheader()
    return writer, fh


def _load_completed(path: str) -> set:
    """Return set of (agent_name, clone_name, clone_role, trial, aligned,
    agent_score_input) already in CSV.

    clone_name is normalised (no .npz) to match how trial_key is built.
    Old CSVs without 'aligned' default to True; without 'agent_score_input'
    default to the clone_score value (pre-robustness behaviour).
    """
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clone_name = row.get("clone_name", "").replace(".npz", "")
            done.add((
                row.get("agent_name", ""),
                clone_name,
                row.get("clone_role", ""),
                row.get("trial", ""),
                row.get("aligned", "True"),
                row.get("agent_score_input", row.get("clone_score", "")),
            ))
    return done


# ── CCM model selector ────────────────────────────────────────────────────────

class CCMAgent:
    """
    Trust-Driven Model agent. Selects which checkpoint to use based on the
    clone's score for the metric at evaluation time.
    """
    def __init__(
        self,
        low_path:     Optional[str],
        mid_path:     Optional[str],
        high_path:    Optional[str],
        obs_dim:      int,
        hidden_sizes: Tuple[int, ...],
        Q1:           float,
        Q2:           float,
    ):
        self.obs_dim      = obs_dim
        self.hidden_sizes = hidden_sizes
        self.Q1 = Q1
        self.Q2 = Q2

        self.params = {}
        for label, path in [("Low", low_path), ("Middle", mid_path), ("High", high_path)]:
            if path and os.path.exists(path):
                self.params[label] = load_agent_params(path, obs_dim, hidden_sizes)
            else:
                self.params[label] = None

    def get_params(self, score: float) -> Optional[object]:
        group = score_to_group(score, self.Q1, self.Q2)
        return self.params.get(group)

    def available(self) -> bool:
        return any(p is not None for p in self.params.values())


# ── per-script evaluation ─────────────────────────────────────────────────────

def evaluate_script(
    script_path:  str,
    models_dir:   str,
    clones_dir:   str,
    clone_scores: Dict,
    train_sets:   Dict,
    out_dir:      str,
    n_trials:     int,
    max_steps:    int,
    cook_time:    int,
    hidden_sizes: Tuple[int, ...],
    seed:         int,
    overwrite:    bool,
    verbose:      bool,
    baselines_dir: str  = "baselines/",
    robustness_sweep: bool = False,
    include_unaligned_trials: bool = False,
):
    # include_unaligned_trials: for each (clone, clone_role), the clone's
    # score is looked up under that role (e.g. clone_entry["clone_0"]).
    # That score is only meaningful if the clone is ACTUALLY seated in the
    # matching seat during the episode. By default, all n_trials episodes
    # for a given clone_role place the clone in that exact seat, so
    # clone_score always matches clone_seat. Setting this True additionally
    # runs n_trials MORE episodes with the clone forced into the OTHER
    # seat instead — i.e. deliberately using a score that does NOT match
    # where the clone is actually playing. Those rows are still written
    # (marked via the "aligned" column) in case that mismatch condition is
    # itself of interest, but they roughly double total runtime and aren't
    # informative for the ordinary "does this agent adapt to its partner"
    # question, so they're off unless explicitly requested.
    script_name = Path(script_path).stem
    with open(script_path) as f:
        agents_cfg = json.load(f)

    if not agents_cfg:
        print(f"[{script_name}] empty script, skipping")
        return

    # infer layout and metric from script
    layout_raw = agents_cfg[0]["layout"]
    layout     = LAYOUT_NAME_MAP.get(layout_raw, layout_raw)
    layout_key = LAYOUT_TO_TRAIN_KEY.get(layout, layout_raw)

    # infer metric: first agent with non-empty extra_features
    metric = None
    for cfg in agents_cfg:
        ef = cfg.get("extra_features", [])
        if ef:
            metric = ef[0]
            break

    print(f"\n[{script_name}] layout={layout}  metric={metric or 'none'}")

    # test clones (flat list of filenames)
    test_clones = train_sets.get(layout_key, {}).get("Test", [])
    if not test_clones:
        print(f"[{script_name}] WARNING: no test clones found for {layout_key}")
        return

    # thresholds
    Q1 = Q2 = None
    if metric:
        try:
            Q1, Q2 = compute_thresholds(train_sets, layout_key, metric, clone_scores)
            print(f"[{script_name}] thresholds: Q1={Q1:.4f}  Q2={Q2:.4f}")
        except ValueError as e:
            print(f"[{script_name}] WARNING: {e}")

    # ── discover which agents exist ───────────────────────────────────────────
    # build agent list: each entry is
    # (agent_name, agent_type, params_or_None, obs_dim, extra_values_fn)
    # extra_values_fn(clone_score) → List[float]

    agent_entries = []
    missing       = []

    for cfg in agents_cfg:
        # Specialists are evaluated through the CCM/CCM+CF selectors.
        if cfg["output_name"] != f"{layout_key}_CF":
            continue
        name     = cfg["output_name"]
        path     = _model_path(models_dir, name)
        ef       = cfg.get("extra_features", [])
        obs_dim  = FEATURE_DIM + len(ef)

        if not os.path.exists(path):
            missing.append(name)
            agent_entries.append((name, "standard", None, obs_dim, ef))
            continue

        params = load_agent_params(path, obs_dim, hidden_sizes)

        # Stats this specific agent was trained with (from its own training
        # pool — see agent_population.py). Empty dict for agents with no
        # extra_features, or if this agent predates this field.
        feat_stats = load_extra_feature_stats(models_dir, name)

        def _extra_fn(score, _ef=ef, _metric=metric, _stats=feat_stats):
            # extra_values is [z-scored score] if metric in ef, else []
            if _ef and _metric and _metric in _ef:
                if score is None:
                    return [0.0]
                return [_zscore(score, _metric, _stats)]
            return []

        agent_entries.append((name, "standard", params, obs_dim, _extra_fn))

    # CCM agents (if metric available)
    if metric and Q1 is not None:
        layout_prefix = layout_key  # e.g. "Cramped_Room"

        for ccm_name, use_cf in [
            (f"{layout_prefix}_CCM",     False),
            (f"{layout_prefix}_CCM+CF", True),
        ]:
            configuration = "CCM+CF" if use_cf else "CCM"
            low_path  = _model_path(models_dir, f"{layout_prefix}_{configuration}_low")
            mid_path  = _model_path(models_dir, f"{layout_prefix}_{configuration}_middle")
            high_path = _model_path(models_dir, f"{layout_prefix}_{configuration}_high")

            ccm_obs_dim = FEATURE_DIM + (1 if use_cf else 0)
            ccm = CCMAgent(low_path, mid_path, high_path,
                           ccm_obs_dim, hidden_sizes, Q1, Q2)

            for label, path in [("low", low_path), ("middle", mid_path), ("high", high_path)]:
                if not os.path.exists(path):
                    missing.append(f"{layout_prefix}_{configuration}_{label}")

            # CCM+CF routes to a DIFFERENT specialist per score, and each
            # specialist (CCM+CF low/middle/high) was trained with its own
            # feature_stats (each has its own, narrower training pool) — so
            # the correct normalization depends on which one ends up
            # selected, not a single global stat.
            ccm_stats_by_group = {
                "Low":    load_extra_feature_stats(models_dir, f"{layout_prefix}_{configuration}_low"),
                "Middle": load_extra_feature_stats(models_dir, f"{layout_prefix}_{configuration}_middle"),
                "High":   load_extra_feature_stats(models_dir, f"{layout_prefix}_{configuration}_high"),
            }

            def _ccm_extra_fn(score, _use_cf=use_cf, _metric=metric,
                               _stats_by_group=ccm_stats_by_group,
                               _q1=Q1, _q2=Q2):
                if _use_cf and score is not None:
                    group = score_to_group(score, _q1, _q2)
                    stats = _stats_by_group.get(group, {})
                    return [_zscore(score, _metric, stats)]
                return []

            agent_entries.append((ccm_name, "CCM", ccm, ccm_obs_dim, _ccm_extra_fn))

    # ── true baseline (self-play baseline from baselines/) ────────────────────
    baseline_path = os.path.join(baselines_dir, f"{layout}.npz")
    baseline_name = "Baseline"
    if os.path.exists(baseline_path):
        baseline_params = load_agent_params(baseline_path, FEATURE_DIM, hidden_sizes)
        agent_entries.append((
            baseline_name, "Baseline", baseline_params,
            FEATURE_DIM, lambda score: [],
        ))
        if verbose:
            print(f"[{script_name}] Baseline loaded: {baseline_path}")
    else:
        missing.append(f"Baseline ({baseline_path})")
        agent_entries.append((
            baseline_name, "Baseline", None,
            FEATURE_DIM, lambda score: [],
        ))

    # warn about missing models
    if missing:
        print(f"\n[{script_name}] WARNING — the following models were NOT found "
              f"and will be skipped:")
        for m in missing:
            print(f"  {_model_path(models_dir, m)}")

    # ── output files ──────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    results_path     = os.path.join(out_dir, f"{script_name}_results.csv")
    transitions_path = os.path.join(out_dir, f"{script_name}_transitions.csv")

    completed = _load_completed(results_path) if not overwrite else set()

    res_writer,  res_fh  = _open_csv(results_path,     RESULT_COLS,     overwrite)
    trans_writer, trans_fh = _open_csv(transitions_path, TRANSITION_COLS, overwrite)

    rng = np.random.default_rng(seed)

    env = make("overcooked", layout=overcooked_layouts[layout])

    # load clone model once per clone (reuse across trials/roles/agents)
    clone_params_cache: Dict[str, object] = {}

    def _get_clone_params(clone_file: str) -> object:
        if clone_file not in clone_params_cache:
            path = os.path.join(clones_dir, clone_file)
            if not os.path.exists(path):
                clone_params_cache[clone_file] = None
            else:
                like = init_model(FEATURE_DIM, ACT_DIM, hidden_sizes=hidden_sizes,
                                  key=jax.random.PRNGKey(0))
                clone_params_cache[clone_file] = load_params_npz(path, like_params=like)
        return clone_params_cache[clone_file]

    _, init_state = env.reset(jax.random.PRNGKey(seed))
    stations      = _get_layout_stations(env)
    wall_map      = np.array(init_state.wall_map)

    n_score_inputs = 7 if robustness_sweep else 1
    n_passes       = 1 if robustness_sweep else (2 if include_unaligned_trials else 1)
    total_trials   = (len(agent_entries) * len(test_clones) * 2
                      * n_trials * n_passes * n_score_inputs)
    done_count = 0

    for agent_name, agent_type, agent_obj, obs_dim, extra_fn in agent_entries:
        # skip if no model available
        if agent_obj is None:
            if verbose:
                print(f"  [{agent_name}] skipping — model not found")
            continue
        if agent_type == "CCM" and not agent_obj.available():
            if verbose:
                print(f"  [{agent_name}] skipping — no CCM sub-models found")
            continue

        if verbose:
            suffix = " x 2 (aligned + unaligned)" if include_unaligned_trials else ""
            print(f"\n  [{agent_name}] evaluating "
                  f"{len(test_clones)} clones × 2 roles × {n_trials} trials{suffix}")

        for clone_file in test_clones:
            clone_key_name = clone_file.replace(".npz", "")
            clone_entry    = clone_scores.get(clone_key_name, {})
            clone_params   = _get_clone_params(clone_file)

            if clone_params is None:
                print(f"    WARNING: clone {clone_file} not found, skipping")
                continue

            for clone_role in ("clone_0", "clone_1"):
                # score for this clone in this role
                clone_score = None
                if metric:
                    clone_score = clone_entry.get(clone_role, {}).get(metric)

                clone_group = None
                if clone_score is not None and Q1 is not None:
                    clone_group = score_to_group(clone_score, Q1, Q2)

                # clone never has extra features (FEATURE_DIM obs)
                clone_extra = []

                # clone_role determines which seat the clone actually plays.
                # clone_role="clone_0" -> clone seated at 0, agent at 1.
                # clone_role="clone_1" -> clone seated at 1, agent at 0.
                clone_seat_for_role = 0 if clone_role == "clone_0" else 1
                aligned_agent_seat  = 1 - clone_seat_for_role

                # ── determine which score inputs to sweep ──────────────────
                # In robustness mode: always sweep 1-7 regardless of alignment.
                # In normal mode: use the real clone_score (aligned trials only,
                # plus unaligned if requested).
                if robustness_sweep:
                    score_inputs = list(range(1, 8))  # 1,2,3,4,5,6,7
                    passes_for_sweep = [(aligned_agent_seat, True)]
                else:
                    score_inputs = [clone_score]
                    passes_for_sweep = [(aligned_agent_seat, True)]
                    if include_unaligned_trials:
                        passes_for_sweep.append((clone_seat_for_role, False))

                for agent_score_input in score_inputs:
                    # group based on what score the agent actually receives —
                    # in robustness mode this differs from clone_group
                    agent_group = (
                        score_to_group(agent_score_input, Q1, Q2)
                        if Q1 is not None and agent_score_input is not None
                        else clone_group or ""
                    )
                    for agent_seat, aligned in passes_for_sweep:
                        for trial in range(n_trials):
                            # normalise clone_name to remove .npz for key
                            trial_key = (
                                agent_name,
                                clone_key_name,   # already without .npz
                                clone_role,
                                str(trial),
                                str(aligned),
                                str(agent_score_input),
                            )
                            if trial_key in completed:
                                done_count += 1
                                continue

                            ep_seed = int(rng.integers(0, 2**31))

                            # compute extra values using agent_score_input
                            # (may differ from clone_score in robustness mode)
                            agent_extra = extra_fn(agent_score_input)

                            # resolve params for CCM
                            if agent_type == "CCM":
                                params = agent_obj.get_params(
                                    agent_score_input if agent_score_input is not None
                                    else clone_score or 0.0
                                )
                                if params is None:
                                    if verbose:
                                        print(
                                            f"    [{agent_name}] no model for "
                                            f"score={agent_score_input}, skipping"
                                        )
                                    continue
                            else:
                                params = agent_obj

                            result = run_episode(
                                agent_params = params,
                                clone_params = clone_params,
                                agent_extra  = agent_extra,
                                clone_extra  = clone_extra,
                                agent_seat   = agent_seat,
                                env          = env,
                                stations     = stations,
                                wall_map     = wall_map,
                                cook_time    = cook_time,
                                max_steps    = max_steps,
                                seed         = ep_seed,
                            )

                            transitions = result.pop("_transitions")

                            # write result row
                            row = {
                                "script":            script_name,
                                "agent_name":        agent_name,
                                "agent_type":        agent_type,
                                "layout":            layout,
                                "clone_name":        clone_key_name,
                                "clone_role":        clone_role,
                                "clone_score":       clone_score,
                                "clone_group":       clone_group or "",
                                "agent_score_input": agent_score_input,
                                "agent_group":       agent_group,
                                "trial":             trial,
                                "agent_seat":        agent_seat,
                                "aligned":           aligned,
                                **result,
                            }
                            res_writer.writerow(row)
                            res_fh.flush()

                            # write transition rows (long format)
                            for src_ev in KITCHEN_EVENTS:
                                for dst_ev in KITCHEN_EVENTS:
                                    count = transitions[src_ev][dst_ev]
                                    if count > 0:
                                        trans_writer.writerow({
                                            "script":            script_name,
                                            "agent_name":        agent_name,
                                            "clone_name":        clone_key_name,
                                            "clone_role":        clone_role,
                                            "agent_score_input": agent_score_input,
                                        "agent_group":       agent_group,
                                            "trial":             trial,
                                            "agent_seat":        agent_seat,
                                            "aligned":           aligned,
                                            "from_event":        src_ev,
                                            "to_event":          dst_ev,
                                            "count":             count,
                                        })
                            trans_fh.flush()

                            done_count += 1
                            if verbose:
                                pct = 100 * done_count / max(total_trials, 1)
                                print(
                                    f"    [{agent_name}] clone={clone_key_name} "
                                    f"role={clone_role} trial={trial} "
                                    f"seat={agent_seat} aligned={aligned} "
                                    f"score_input={agent_score_input} "
                                    f"agent_group={agent_group} "
                                    f"reward={result['total_reward']:.0f} "
                                    f"deliveries={result['deliveries']} "
                                    f"({pct:.1f}%)"
                                )

    res_fh.close()
    trans_fh.close()

    print(f"\n[{script_name}] done")
    print(f"  results     → {results_path}")
    print(f"  transitions → {transitions_path}")

    if missing:
        print(f"\n[{script_name}] WARNING — evaluation was incomplete. "
              f"The following models were not found:")
        for m in missing:
            print(f"  {_model_path(models_dir, m)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate competence-aware agents against held-out test clones.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scripts", type=str, nargs="+", required=True,
                    help="One or more training script JSON files "
                         "(e.g. configs/cramped_room.json)")
    ap.add_argument("--models-dir", type=str, required=True,
                    help="Root directory of trained agents "
                         "(--out-dir from run_agent_pipeline.py)")
    ap.add_argument("--clones-dir", type=str, required=True,
                    help="Directory containing clone .npz checkpoints")
    ap.add_argument("--clones-scores", type=str, required=True,
                    help="Path to artifacts/partners/competence_labels.json")
    ap.add_argument("--train-sets", type=str, required=True,
                    help="Path to configs/splits.json")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory for results CSVs")
    ap.add_argument("--n-trials", type=int, default=5,
                    help="Trials per clone per role (half as seat 0, "
                         "half as seat 1) (default: 5)")
    ap.add_argument("--max-steps", type=int, default=400,
                    help="Max timesteps per episode (default: 400)")
    ap.add_argument("--cook-time", type=int, default=20,
                    help="Soup cook time in steps (default: 20)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256],
                    help="Network hidden layer sizes (default: 256 256)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files. Default: continue "
                         "from where evaluation left off.")
    ap.add_argument("--baselines-dir", type=str, default="baselines/",
                    help="Directory containing self-play baseline .npz files "
                         "(default: baselines/)")
    ap.add_argument("--robustness-sweep", action="store_true",
                    help="For every agent × clone × role × trial, sweep the "
                         "agent_score_input across all integer values 1-7 "
                         "(the full metric range). In this mode the aligned "
                         "column is ignored — all trials run regardless of "
                         "alignment. Adds 7x trials per clone/role/agent. "
                         "Useful for isolating model behaviour differences "
                         "from clone score differences.")
    ap.add_argument("--include-unaligned-trials", action="store_true",
                    help="Also run trials where the clone is deliberately "
                         "seated OPPOSITE the role its score was looked up "
                         "under (e.g. using the clone_0 score while the "
                         "clone actually plays seat 1). Off by default: "
                         "roughly doubles runtime and isn't informative for "
                         "the ordinary adaptation question, since clone_score "
                         "then doesn't match how the clone is actually "
                         "playing that episode.")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.clones_scores) as f:
        clone_scores = json.load(f)
    with open(args.train_sets) as f:
        train_sets = json.load(f)

    hidden_sizes = tuple(args.hidden)

    for script_path in args.scripts:
        if not os.path.exists(script_path):
            print(f"[warn] script not found: {script_path} — skipping")
            continue

        evaluate_script(
            script_path      = script_path,
            models_dir       = args.models_dir,
            clones_dir       = args.clones_dir,
            clone_scores     = clone_scores,
            train_sets       = train_sets,
            out_dir          = args.out_dir,
            n_trials         = args.n_trials,
            max_steps        = args.max_steps,
            cook_time        = args.cook_time,
            hidden_sizes     = hidden_sizes,
            seed             = args.seed,
            overwrite        = args.overwrite,
            verbose          = not args.quiet,
            baselines_dir    = args.baselines_dir,
            robustness_sweep = args.robustness_sweep,
            include_unaligned_trials = args.include_unaligned_trials,
        )


if __name__ == "__main__":
    main()