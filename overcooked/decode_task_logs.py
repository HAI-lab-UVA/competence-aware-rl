"""
decode_trial_logs.py — Turn a trial_logs.csv (from evaluate_agents.py's
--save-trial-logs) back into full timestep-by-timestep JSON.

Each trial log row stores only the episode's random seed and the raw
per-step actions taken by each side, zlib-compressed and base64-encoded
(see encode_trial_actions/decode_trial_actions in evaluate_agents.py).
That's enough to exactly recreate the episode: reset the environment with
the same seed, then step through the recorded actions in order — no model
checkpoints needed, since the actions are already known.

This script does that replay and expands each trial into a full step-by-step
record: positions, facing direction, held item, action, reward, and fired
kitchen events for both sides at every timestep.

Usage:
    python decode_trial_logs.py \\
        --trial-logs evaluation/agents/cramped_room_trial_logs.csv \\
        --layout Cramped_Room \\
        --out cramped_room_trial_logs_expanded.jsonl

    # Only a specific trial (much faster than decoding the whole file):
    python decode_trial_logs.py \\
        --trial-logs evaluation/agents/cramped_room_trial_logs.csv \\
        --layout Cramped_Room \\
        --agent-name Cramped_Room_CCM_low --clone-name clone_003 \\
        --clone-role clone_0 --trial 2 \\
        --out one_trial.json

Output: one JSON object per trial, written as JSON Lines (one per line) by
default, or as a single pretty-printed JSON array if --out ends in a name
you pass alongside --single-trial / when only one trial matches.
Each object looks like:
    {
      "trial_meta": {... every column from trial_logs.csv except
                      actions_b64 ...},
      "steps": [
        {
          "step": 0,
          "agent": {"action": 2, "pos": [3, 1], "dir": [0, -1],
                     "held_pre": "empty", "held_post": "empty",
                     "reward": 0.0, "events": []},
          "clone": {...same shape...},
          "collision": false
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

from bc_features import _get_layout_stations
from evaluate_agents import (
    LAYOUT_NAME_MAP,
    decode_trial_actions,
    detect_events_from_inv,
)
from evaluate_population import classify_movement

_INV_TO_HELD = {1: "empty", 3: "onion", 5: "plate", 9: "soup"}


def _held_from_state(state, seat: int) -> str:
    return _INV_TO_HELD.get(int(state.agent_inv[seat]), "empty")


def replay_trial(
    row:        Dict[str, str],
    env,
    stations:   dict,
    cook_time:  int,
) -> Dict:
    """
    Replay a single trial_logs.csv row through the environment and return
    the full timestep-by-timestep record.
    """
    agent_seat = int(row["agent_seat"])
    clone_seat = 1 - agent_seat
    agent_key  = f"agent_{agent_seat}"
    clone_key  = f"agent_{clone_seat}"

    agent_actions, clone_actions = decode_trial_actions(row["actions_b64"])
    total_steps = int(row["total_steps"])
    seed = int(row["episode_seed"])

    # exactly mirrors run_episode()'s own key derivation, so the replay is
    # bit-for-bit identical to the original run.
    key = jax.random.PRNGKey(seed)
    key, k_r = jax.random.split(key)
    obs, state = env.reset(k_r)

    pot_positions     = stations.get("pots", set())
    serving_positions = stations.get("serving_counters", set())

    steps_out: List[Dict] = []

    for step in range(total_steps):
        key, k_ag, k_cl, k_step = jax.random.split(key, 4)  # k_ag/k_cl unused (actions already fixed)

        a_act = int(agent_actions[step])
        c_act = int(clone_actions[step])

        pre_state = state
        pre_held_agent = _held_from_state(pre_state, agent_seat)
        pre_held_clone = _held_from_state(pre_state, clone_seat)

        actions = {
            agent_key: jnp.array(a_act, dtype=jnp.uint32),
            clone_key: jnp.array(c_act, dtype=jnp.uint32),
        }
        key, k_step = jax.random.split(key)
        obs, state, rewards, dones, _ = env.step(k_step, state, actions)

        post_held_agent = _held_from_state(state, agent_seat)
        post_held_clone = _held_from_state(state, clone_seat)

        side_records = {}
        for who, seat, act, pre_held, post_held, o_key in [
            ("agent", agent_seat, a_act, pre_held_agent, post_held_agent, agent_key),
            ("clone", clone_seat, c_act, pre_held_clone, post_held_clone, clone_key),
        ]:
            pos = (int(pre_state.agent_pos[seat][0]), int(pre_state.agent_pos[seat][1]))
            direction = (int(pre_state.agent_dir[seat][0]), int(pre_state.agent_dir[seat][1]))
            reward = float(rewards[o_key])

            evs = detect_events_from_inv(
                act=act, pre_inv=pre_held, post_inv=post_held, reward=reward,
                agent_pos=pos, agent_dir=direction,
                pot_positions=pot_positions, serving_positions=serving_positions,
            )
            fired = [name for name, on in evs.items() if on]

            side_records[who] = {
                "action":    act,
                "pos":       list(pos),
                "dir":       list(direction),
                "held_pre":  pre_held,
                "held_post": post_held,
                "reward":    reward,
                "events":    fired,
            }

        col, _a0_out, _a1_out = classify_movement(
            pre_state, state,
            a_act if agent_seat == 0 else c_act,
            c_act if agent_seat == 0 else a_act,
        )

        steps_out.append({
            "step":      step,
            "agent":     side_records["agent"],
            "clone":     side_records["clone"],
            "collision": bool(col),
        })

        if dones.get("__all__", False):
            break

    trial_meta = {k: v for k, v in row.items() if k != "actions_b64"}
    return {"trial_meta": trial_meta, "steps": steps_out}


def _row_matches(row: Dict[str, str], filters: Dict[str, Optional[str]]) -> bool:
    for field, value in filters.items():
        if value is not None and row.get(field, "") != str(value):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Expand a trial_logs.csv back into timestep-by-timestep JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trial-logs", type=str, required=True,
                     help="Path to the *_trial_logs.csv produced by "
                          "evaluate_agents.py --save-trial-logs")
    ap.add_argument("--layout", type=str, required=True,
                     help="Layout the script was run on, e.g. Cramped_Room, "
                          "Forced_Coordination, asymm_advantages, etc. "
                          "(same names evaluate_agents.py accepts)")
    ap.add_argument("--cook-time", type=int, default=20,
                     help="Soup cook time in steps (must match the original "
                          "evaluate_agents.py run; default: 20)")
    ap.add_argument("--out", type=str, default=None,
                     help="Output path. Default: alongside --trial-logs, "
                          "same stem with _expanded.jsonl. Written as JSON "
                          "Lines (one trial object per line).")
    # optional filters — decoding is one full env replay per trial, so
    # narrowing down which rows to expand is much faster than doing all of them
    ap.add_argument("--agent-name", type=str, default=None)
    ap.add_argument("--clone-name", type=str, default=None)
    ap.add_argument("--clone-role", type=str, default=None)
    ap.add_argument("--agent-score-input", type=str, default=None)
    ap.add_argument("--trial", type=str, default=None)
    ap.add_argument("--agent-seat", type=str, default=None)
    ap.add_argument("--aligned", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None,
                     help="Stop after decoding this many matching trials.")
    ap.add_argument("--pretty", action="store_true",
                     help="Pretty-print each JSON object (still one per "
                          "line won't be valid JSONL if used with multiple "
                          "trials — intended for single-trial extraction).")
    args = ap.parse_args()

    layout = LAYOUT_NAME_MAP.get(args.layout, args.layout)
    if layout not in overcooked_layouts:
        print(f"[error] unknown layout '{args.layout}' -> '{layout}'. "
              f"Known: {sorted(overcooked_layouts.keys())}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(args.trial_logs)[0]
        out_path = f"{stem}_expanded.jsonl"

    env = make("overcooked", layout=overcooked_layouts[layout])
    _, init_state = env.reset(jax.random.PRNGKey(0))
    stations = _get_layout_stations(env)

    filters = {
        "agent_name":        args.agent_name,
        "clone_name":        args.clone_name,
        "clone_role":        args.clone_role,
        "agent_score_input": args.agent_score_input,
        "trial":             args.trial,
        "agent_seat":        args.agent_seat,
        "aligned":           args.aligned,
    }

    n_written = 0
    with open(args.trial_logs, newline="") as f_in, open(out_path, "w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if not _row_matches(row, filters):
                continue

            expanded = replay_trial(row, env, stations, args.cook_time)

            if args.pretty:
                f_out.write(json.dumps(expanded, indent=2))
                f_out.write("\n")
            else:
                f_out.write(json.dumps(expanded))
                f_out.write("\n")

            n_written += 1
            if args.limit is not None and n_written >= args.limit:
                break

    print(f"Decoded {n_written} trial(s) → {out_path}")
    if n_written == 0:
        print("[warn] no rows matched your filters — check "
              "--agent-name/--clone-name/--clone-role/--trial/etc.")


if __name__ == "__main__":
    main()