"""
evaluate_baselines.py — Evaluate self-play baseline models across all layouts.

For each baseline found in --baselines-dir, runs N episodes of self-play
(model vs frozen copy of itself, alternating seats) and records:

  Performance:
    - Total reward and deliveries per episode
    - Mean / min time between deliveries
    - Time to first delivery
    - Time to first onion pickup, first onion in pot, first plate pickup

  Per-agent kitchen events (agent_0 = red, agent_1 = blue):
    pick_onion, place_onion_counter, place_onion_pot,
    pick_plate, place_plate_counter, fill_plate,
    place_soup_counter, deliver_soup

  Per-agent step types:
    idle_steps, move_ok_steps, interact_steps,
    counter_bumps, agent_bumps

  Collisions:
    Steps where agents blocked / contested the same cell

Results are written to:
  baseline_evaluation_results.csv   — one row per layout × episode
  baseline_evaluation_summary.csv   — one row per layout (episode averages)
  baseline_evaluation_summary.json  — same, for programmatic use

Usage:
    python evaluate_baselines.py \\
        --baselines-dir artifacts/baselines/ \\
        --out-dir       evaluation/baselines/ \\
        --n-episodes    10 \\
        --max-steps     400
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

from bc_features import featurize_jaxmarl, FEATURE_DIM, _get_layout_stations
from network import init_model, forward
from checkpoint import load_params_npz

# Re-use event detection machinery from evaluate_population
from evaluate_population import (
    detect_agent_events,
    classify_movement,
    KITCHEN_EVENTS,
    MOVE_DELTAS,
    MOVE_ACTIONS,
    STAY_ACTION,
    INTERACT_ACTION,
    ACTION_NAMES,
    CH_ONION, CH_DISH, CH_SOUP,
    CH_POT, CH_POT_ONIONS, CH_POT_COOKING, CH_COUNTER,
    INV_EMPTY, INV_ONION, INV_PLATE, INV_SOUP,
    _EV_ABBREV,
    _rule, _divider, _f,
)

ACT_DIM = 6

ALL_LAYOUTS = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
]


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(
    params,
    env,
    stations,
    wall_map:    np.ndarray,
    cook_time:   int,
    max_steps:   int,
    learner_seat: int,
    key,
    debug_events: bool = False,
) -> Dict:
    """
    Run one self-play episode. The model occupies both seats — learner_seat
    determines which agent's reward is tracked as 'primary'. The partner uses
    identical params (frozen copy).

    Returns a flat dict of per-episode metrics.
    """
    partner_seat = 1 - learner_seat
    learner_key  = f"agent_{learner_seat}"
    partner_key  = f"agent_{partner_seat}"

    key, k_r = jax.random.split(key)
    obs, state = env.reset(k_r)

    # per-agent event counters
    events_a = [{e: 0 for e in KITCHEN_EVENTS} for _ in range(2)]
    held      = ["empty", "empty"]

    # step-type counters
    idle_steps     = [0, 0]
    move_ok_steps  = [0, 0]
    interact_steps = [0, 0]
    counter_bumps  = [0, 0]
    agent_bumps    = [0, 0]

    # performance tracking
    total_reward        = 0.0
    deliveries          = 0
    delivery_times:     List[int] = []
    first_onion_pickup  = None
    first_onion_pot     = None
    first_plate_pickup  = None

    prev_obs = {
        learner_key: np.array(obs[learner_key]),
        partner_key: np.array(obs[partner_key]),
    }

    collisions = 0
    n_steps    = 0

    for step in range(max_steps):
        key, k_l, k_p, k_step = jax.random.split(key, 4)

        pre_state = state
        actions   = {}
        chosen    = {}

        for seat, (agent_key, k_act) in enumerate([
            (learner_key, k_l), (partner_key, k_p)
        ]):
            feat = featurize_jaxmarl(
                obs        = np.array(obs[agent_key]),
                state      = state,
                player_idx = seat,
                stations   = stations,
                wall_map   = wall_map,
                grid_w     = env.width,
                grid_h     = env.height,
                cook_time  = cook_time,
            )
            logits, _ = forward(params, jnp.asarray(feat))
            action    = int(jax.random.categorical(k_act, logits))
            actions[agent_key] = jnp.array(action, dtype=jnp.uint32)
            chosen[seat]       = action

        key, k_step = jax.random.split(key)
        obs, state, rewards, dones, _ = env.step(k_step, state, actions)
        n_steps = step + 1

        # ── movement classification (uses pre/post state.agent_pos) ───────────
        collision, a0_outcome, a1_outcome = classify_movement(
            pre_state, state, chosen[0], chosen[1]
        )
        if collision:
            collisions += 1

        outcomes = [a0_outcome, a1_outcome]
        for seat in range(2):
            action  = chosen[seat]
            outcome = outcomes[seat]

            if action == STAY_ACTION:
                idle_steps[seat] += 1
            elif action == INTERACT_ACTION:
                interact_steps[seat] += 1
            else:
                if outcome == "move_ok":
                    move_ok_steps[seat] += 1
                elif outcome == "ctr_bump":
                    counter_bumps[seat] += 1
                elif outcome == "agent_bump":
                    agent_bumps[seat] += 1

        # ── per-agent kitchen event detection ─────────────────────────────────
        for seat in range(2):
            agent_key = f"agent_{seat}"
            action    = chosen[seat]
            raw_r     = float(rewards[agent_key])
            curr_obs  = np.array(obs[agent_key])
            p_obs     = prev_obs[agent_key]

            d_obs                = curr_obs - p_obs
            d_onion              = float(d_obs[:, :, CH_ONION].sum())
            d_dish               = float(d_obs[:, :, CH_DISH].sum())
            pot_mask             = p_obs[:, :, CH_POT] > 0.5
            d_pot_onions_at_pot  = float((d_obs[:, :, CH_POT_ONIONS]  * pot_mask).sum())
            d_pot_cooking_at_pot = float((d_obs[:, :, CH_POT_COOKING] * pot_mask).sum())
            d_soup_at_pot        = float((d_obs[:, :, CH_SOUP]        * pot_mask).sum())
            d_counter            = float(d_obs[:, :, CH_COUNTER].sum())

            evs, new_held = detect_agent_events(
                action               = action,
                held                 = held[seat],
                reward               = raw_r,
                d_onion              = d_onion,
                d_dish               = d_dish,
                d_pot_onions_at_pot  = d_pot_onions_at_pot,
                d_pot_cooking_at_pot = d_pot_cooking_at_pot,
                d_soup_at_pot        = d_soup_at_pot,
                d_counter            = d_counter,
            )
            held[seat] = new_held
            for ev, fired in evs.items():
                if fired:
                    events_a[seat][ev] += 1

            prev_obs[agent_key] = curr_obs

        # ── primary agent (learner_seat) performance ──────────────────────────
        raw_r = float(rewards[learner_key])
        total_reward += raw_r
        if raw_r > 0:
            deliveries += 1
            delivery_times.append(step)

        # milestone tracking (first occurrences)
        if first_onion_pickup is None and events_a[learner_seat]["pick_onion"] > 0:
            first_onion_pickup = step
        if first_onion_pot is None and events_a[learner_seat]["place_onion_pot"] > 0:
            first_onion_pot = step
        if first_plate_pickup is None and events_a[learner_seat]["pick_plate"] > 0:
            first_plate_pickup = step

        if dones.get("__all__", False):
            break

    # ── delivery timing ───────────────────────────────────────────────────────
    time_to_first = delivery_times[0] if delivery_times else None
    if len(delivery_times) > 1:
        gaps     = [delivery_times[i+1] - delivery_times[i]
                    for i in range(len(delivery_times) - 1)]
        mean_gap = float(np.mean(gaps))
        min_gap  = float(np.min(gaps))
    else:
        mean_gap = min_gap = None

    result = {
        "learner_seat":                learner_seat,
        "total_reward":                total_reward,
        "deliveries":                  deliveries,
        "time_to_first_delivery":      time_to_first,
        "mean_gap_between_deliveries": mean_gap,
        "min_gap_between_deliveries":  min_gap,
        "first_onion_pickup":          first_onion_pickup,
        "first_onion_pot":             first_onion_pot,
        "first_plate_pickup":          first_plate_pickup,
        "collisions":                  collisions,
        "steps":                       n_steps,
    }

    for seat in range(2):
        result[f"a{seat}_idle_steps"]     = idle_steps[seat]
        result[f"a{seat}_move_ok_steps"]  = move_ok_steps[seat]
        result[f"a{seat}_interact_steps"] = interact_steps[seat]
        result[f"a{seat}_counter_bumps"]  = counter_bumps[seat]
        result[f"a{seat}_agent_bumps"]    = agent_bumps[seat]
        for ev in KITCHEN_EVENTS:
            result[f"a{seat}_{ev}"] = events_a[seat][ev]

    return result, key


# ── per-layout evaluation ─────────────────────────────────────────────────────

def evaluate_baseline(
    checkpoint:   str,
    layout:       str,
    n_episodes:   int,
    max_steps:    int,
    hidden_sizes: Tuple[int, ...],
    cook_time:    int,
    seed:         int,
    verbose:      bool,
    debug_events: bool,
) -> List[Dict]:
    """
    Evaluate one baseline checkpoint on one layout.
    Episodes alternate learner seat (even=0, odd=1).
    Returns a list of per-episode result dicts.
    """
    like   = init_model(FEATURE_DIM, ACT_DIM, hidden_sizes=hidden_sizes)
    params = load_params_npz(checkpoint, like_params=like)

    env      = make("overcooked", layout=overcooked_layouts[layout])
    key      = jax.random.PRNGKey(seed)
    key, k_r = jax.random.split(key)
    _, state = env.reset(k_r)
    stations = _get_layout_stations(env)
    wall_map = np.array(state.wall_map)

    if verbose:
        print(f"[eval baseline] layout={layout}  episodes={n_episodes}")

    results = []
    for ep in range(n_episodes):
        learner_seat = ep % 2
        ep_result, key = run_episode(
            params       = params,
            env          = env,
            stations     = stations,
            wall_map     = wall_map,
            cook_time    = cook_time,
            max_steps    = max_steps,
            learner_seat = learner_seat,
            key          = key,
            debug_events = debug_events,
        )
        ep_result["layout"]     = layout
        ep_result["episode"]    = ep
        ep_result["checkpoint"] = os.path.basename(checkpoint)
        results.append(ep_result)

        if verbose:
            print(
                f"  ep {ep:3d}  seat={learner_seat}  "
                f"deliveries={ep_result['deliveries']}  "
                f"reward={ep_result['total_reward']:.1f}  "
                f"steps={ep_result['steps']}"
            )

    return results


# ── summarise ─────────────────────────────────────────────────────────────────

def summarise(results: List[Dict]) -> List[Dict]:
    """Aggregate per-episode results into per-layout summaries."""
    from collections import defaultdict
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        grouped[r["layout"]].append(r)

    summaries = []
    for layout in ALL_LAYOUTS:
        eps = grouped.get(layout, [])
        if not eps:
            continue

        def _mean(key):
            vals = [e[key] for e in eps if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        s = {
            "layout":                    layout,
            "n_episodes":                len(eps),
            "mean_total_reward":         _mean("total_reward"),
            "mean_deliveries":           _mean("deliveries"),
            "mean_time_to_first_delivery": _mean("time_to_first_delivery"),
            "mean_gap_between_deliveries": _mean("mean_gap_between_deliveries"),
            "min_gap_between_deliveries":  _mean("min_gap_between_deliveries"),
            "mean_first_onion_pickup":     _mean("first_onion_pickup"),
            "mean_first_onion_pot":        _mean("first_onion_pot"),
            "mean_first_plate_pickup":     _mean("first_plate_pickup"),
            "mean_collisions":             _mean("collisions"),
            "mean_steps":                  _mean("steps"),
        }

        for seat in range(2):
            s[f"mean_a{seat}_idle_steps"]     = _mean(f"a{seat}_idle_steps")
            s[f"mean_a{seat}_move_ok_steps"]  = _mean(f"a{seat}_move_ok_steps")
            s[f"mean_a{seat}_interact_steps"] = _mean(f"a{seat}_interact_steps")
            s[f"mean_a{seat}_counter_bumps"]  = _mean(f"a{seat}_counter_bumps")
            s[f"mean_a{seat}_agent_bumps"]    = _mean(f"a{seat}_agent_bumps")
            for ev in KITCHEN_EVENTS:
                s[f"mean_a{seat}_{ev}"] = _mean(f"a{seat}_{ev}")

        summaries.append(s)

    return summaries


# ── CSV / JSON writers ────────────────────────────────────────────────────────

def _episode_columns() -> List[str]:
    cols = [
        "checkpoint", "layout", "episode", "learner_seat",
        "total_reward", "deliveries", "steps",
        "time_to_first_delivery",
        "mean_gap_between_deliveries", "min_gap_between_deliveries",
        "first_onion_pickup", "first_onion_pot", "first_plate_pickup",
        "collisions",
    ]
    for seat in range(2):
        cols += [
            f"a{seat}_idle_steps", f"a{seat}_move_ok_steps",
            f"a{seat}_interact_steps", f"a{seat}_counter_bumps",
            f"a{seat}_agent_bumps",
        ]
        for ev in KITCHEN_EVENTS:
            cols.append(f"a{seat}_{ev}")
    return cols


def _summary_columns() -> List[str]:
    cols = ["layout", "n_episodes",
            "mean_total_reward", "mean_deliveries", "mean_steps",
            "mean_time_to_first_delivery",
            "mean_gap_between_deliveries", "min_gap_between_deliveries",
            "mean_first_onion_pickup", "mean_first_onion_pot",
            "mean_first_plate_pickup", "mean_collisions"]
    for seat in range(2):
        cols += [
            f"mean_a{seat}_idle_steps", f"mean_a{seat}_move_ok_steps",
            f"mean_a{seat}_interact_steps", f"mean_a{seat}_counter_bumps",
            f"mean_a{seat}_agent_bumps",
        ]
        for ev in KITCHEN_EVENTS:
            cols.append(f"mean_a{seat}_{ev}")
    return cols


def write_episode_csv(results: List[Dict], path: str):
    cols = _episode_columns()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[eval] episode CSV -> {path}")


def write_summary_csv(summaries: List[Dict], path: str):
    cols = _summary_columns()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"[eval] summary CSV -> {path}")


def write_summary_json(summaries: List[Dict], path: str):
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[eval] summary JSON -> {path}")


# ── console tables ────────────────────────────────────────────────────────────

def print_performance_table(summaries: List[Dict]):
    LW = 22
    total = LW + 52
    header = (
        f"  {'layout':<{LW}}  "
        f"{'deliveries':>10}  "
        f"{'reward':>8}  "
        f"{'1st_del':>7}  "
        f"{'gap_mean':>8}  "
        f"{'collide':>7}"
    )
    print(f"\n  PERFORMANCE  (mean / episode · self-play both seats)")
    print(f"  {_rule(total)}")
    print(header)
    print(f"  {_rule(total)}")
    for s in summaries:
        print(
            f"  {s['layout']:<{LW}}  "
            f"{_f(s.get('mean_deliveries')):>10}  "
            f"{_f(s.get('mean_total_reward')):>8}  "
            f"{_f(s.get('mean_time_to_first_delivery')):>7}  "
            f"{_f(s.get('mean_gap_between_deliveries')):>8}  "
            f"{_f(s.get('mean_collisions')):>7}"
        )
    print(f"  {_rule(total)}")


def print_steps_table(summaries: List[Dict]):
    LW = 22
    AW = 5
    IW = 7
    MW = 8
    TW = 9
    BW = 8
    GW = 9
    total = LW + AW + IW + MW + TW + BW + GW + 14

    header = (
        f"  {'layout':<{LW}}  {'agent':<{AW}}  "
        f"{'idle':>{IW}}  {'move_ok':>{MW}}  "
        f"{'interact':>{TW}}  {'ctr_bump':>{BW}}  {'agt_bump':>{GW}}"
    )
    print(f"\n  TIME ALLOCATION  (mean steps / episode · agent_0=red · agent_1=blue)")
    print(f"  {_rule(total)}")
    print(header)
    print(f"  {_rule(total)}")

    for s in summaries:
        for ag, label in ((0, "red"), (1, "blue")):
            layout_col = f"{s['layout']:<{LW}}" if ag == 0 else " " * LW
            print(
                f"  {layout_col}  {label:<{AW}}  "
                f"{_f(s.get(f'mean_a{ag}_idle_steps')):>{IW}}  "
                f"{_f(s.get(f'mean_a{ag}_move_ok_steps')):>{MW}}  "
                f"{_f(s.get(f'mean_a{ag}_interact_steps')):>{TW}}  "
                f"{_f(s.get(f'mean_a{ag}_counter_bumps')):>{BW}}  "
                f"{_f(s.get(f'mean_a{ag}_agent_bumps')):>{GW}}"
            )
        print(f"  {_divider(total)}")
    print(f"  {_rule(total)}")


def print_events_table(summaries: List[Dict]):
    abbrevs = [_EV_ABBREV[e] for e in KITCHEN_EVENTS]
    EVC = 5
    LW  = 22
    AW  = 5
    ev_block = len(abbrevs) * (EVC + 2) - 2
    total    = LW + AW + ev_block + 6

    legend_parts = [f"{v}={k}" for k, v in _EV_ABBREV.items()]
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
        f"  {'layout':<{LW}}  {'agent':<{AW}}  "
        + "  ".join(f"{a:>{EVC}}" for a in abbrevs)
    )
    print(f"\n  KITCHEN EVENTS  (mean count / episode · agent_0=red · agent_1=blue)")
    for ll in legend_lines:
        print(ll)
    print(f"  {_rule(total)}")
    print(header)
    print(f"  {_rule(total)}")

    for s in summaries:
        for ag, label in ((0, "red"), (1, "blue")):
            layout_col = f"{s['layout']:<{LW}}" if ag == 0 else " " * LW
            ev_cols    = "  ".join(
                f"{_f(s.get(f'mean_a{ag}_{e}')):>{EVC}}"
                for e in KITCHEN_EVENTS
            )
            print(f"  {layout_col}  {label:<{AW}}  {ev_cols}")
        print(f"  {_divider(total)}")
    print(f"  {_rule(total)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate self-play baseline models across all layouts."
    )
    ap.add_argument("--baselines-dir", type=str, default="baselines/",
                    help="Directory containing baseline .npz files "
                         "(default: baselines/)")
    ap.add_argument("--out-dir",       type=str, default="evaluation/baselines/",
                    help="Output directory for CSV and JSON results "
                         "(default: evaluation/baselines/)")
    ap.add_argument("--n-episodes",    type=int, default=10,
                    help="Episodes per layout (default: 10)")
    ap.add_argument("--max-steps",     type=int, default=400,
                    help="Max steps per episode (default: 400)")
    ap.add_argument("--hidden",        type=int, nargs="+", default=[256, 256],
                    help="Network hidden layer sizes (default: 256 256)")
    ap.add_argument("--cook-time",     type=int, default=20,
                    help="Soup cook time in steps (default: 20)")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--layout",        type=str, default=None,
                    help="Evaluate only this layout (default: all present)")
    ap.add_argument("--quiet",         action="store_true")
    ap.add_argument("--debug-events",  action="store_true",
                    help="Print detailed channel deltas on every interact step "
                         "(very verbose; use with --n-episodes 1 --layout X)")
    return ap.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    layouts_to_eval = [args.layout] if args.layout else ALL_LAYOUTS

    # find which baselines actually exist
    available = []
    missing   = []
    for layout in layouts_to_eval:
        path = os.path.join(args.baselines_dir, f"{layout}.npz")
        if os.path.exists(path):
            available.append((layout, path))
        else:
            missing.append(layout)

    if missing:
        print(f"[eval] WARNING: no baseline found for: {missing}")
    if not available:
        print(f"[eval] nothing to evaluate — exiting")
        return

    print(f"[eval] baselines_dir: {args.baselines_dir}")
    print(f"[eval] evaluating {len(available)} layout(s) × "
          f"{args.n_episodes} episodes each")
    print(f"[eval] agent_0 = red chef  |  agent_1 = blue chef")

    os.makedirs(args.out_dir, exist_ok=True)

    all_results: List[Dict] = []
    start = time.time()

    for layout, checkpoint in available:
        results = evaluate_baseline(
            checkpoint   = checkpoint,
            layout       = layout,
            n_episodes   = args.n_episodes,
            max_steps    = args.max_steps,
            hidden_sizes = tuple(args.hidden),
            cook_time    = args.cook_time,
            seed         = args.seed,
            verbose      = not args.quiet,
            debug_events = args.debug_events,
        )
        all_results.extend(results)

    summaries = summarise(all_results)

    write_episode_csv(
        all_results,
        os.path.join(args.out_dir, "baseline_evaluation_results.csv"),
    )
    write_summary_csv(
        summaries,
        os.path.join(args.out_dir, "baseline_evaluation_summary.csv"),
    )
    write_summary_json(
        summaries,
        os.path.join(args.out_dir, "baseline_evaluation_summary.json"),
    )

    print_performance_table(summaries)
    print_steps_table(summaries)
    print_events_table(summaries)

    elapsed = time.time() - start
    print(f"\n[eval] done in {elapsed:.1f}s  —  results in {args.out_dir}")


if __name__ == "__main__":
    main()