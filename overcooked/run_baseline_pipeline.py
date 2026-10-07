"""
run_baseline_pipeline.py — Train and validate self-play baseline models for
all 5 Overcooked layouts.

For each layout:
  1. Check if a validated baseline already exists at baselines/<layout>.npz —
     skip if so.
  2. Train a baseline via PBT self-play (baseline_population.py).
  3. Validate the best checkpoint by checking mean soup deliveries per episode
     against a per-layout threshold.
  4. On success, save to baselines/<layout>.npz.
  5. On failure, retry up to --max-retries times with a different seed.
     If all retries fail, log a warning and continue to the next layout
     (do NOT save a partial checkpoint — the agent pipeline will fail fast
     when it cannot find the baseline for that layout).

Layouts are referenced by their JaxMARL internal names:
    cramped_room, asymm_advantages, coord_ring, counter_circuit, forced_coord

Per-layout delivery thresholds default to:
    cramped_room:      3
    asymm_advantages:  2
    coord_ring:        2
    counter_circuit:   1
    forced_coord:      1

Both --total-updates and --min-deliveries accept either a plain number
(uniform across all layouts) or a JSON dict of per-layout overrides, e.g.:
    --total-updates '{"cramped_room": 2000, "forced_coord": 500}'
    --min-deliveries '{"cramped_room": 4, "forced_coord": 2}'

Usage:
    python run_baseline_pipeline.py --baselines-dir artifacts/baselines/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Union

from baseline_population import train_baseline

# ── constants ─────────────────────────────────────────────────────────────────

# Internal JaxMARL names — these are the canonical names used for filenames
# and passed to the environment. agents.json may use human-readable aliases
# which LAYOUT_NAME_MAP (in bc_features.py) converts to these.
ALL_LAYOUTS = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
]

DEFAULT_MIN_DELIVERIES: Dict[str, float] = {
    "cramped_room":     3.0,
    "asymm_advantages": 2.0,
    "coord_ring":       2.0,
    "counter_circuit":  1.0,
    "forced_coord":     1.0,
}

DEFAULT_TOTAL_UPDATES: Dict[str, int] = {
    "cramped_room":     1000,
    "asymm_advantages": 1000,
    "coord_ring":       1000,
    "counter_circuit":  1000,
    "forced_coord":     1000,
}


# ── argument parsing helpers ──────────────────────────────────────────────────

def _parse_per_layout_float(value: str, default_map: Dict[str, float]) -> Dict[str, float]:
    """
    Parse a CLI value that can be either:
      - a plain float/int  → applied uniformly to all layouts
      - a JSON dict        → per-layout overrides; missing layouts use default_map
    """
    result = dict(default_map)
    if value is None:
        return result
    try:
        uniform = float(value)
        return {k: uniform for k in ALL_LAYOUTS}
    except ValueError:
        overrides = json.loads(value)
        result.update({k: float(v) for k, v in overrides.items()})
        return result


def _parse_per_layout_int(value: str, default_map: Dict[str, int]) -> Dict[str, int]:
    """Same as above but for int values (e.g. total_updates)."""
    result = dict(default_map)
    if value is None:
        return result
    try:
        uniform = int(value)
        return {k: uniform for k in ALL_LAYOUTS}
    except ValueError:
        overrides = json.loads(value)
        result.update({k: int(v) for k, v in overrides.items()})
        return result


# ── state management ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(state_path: str) -> Dict:
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


def _save_state(state: Dict, state_path: str):
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


def _init_state(state_path: str, config: Dict) -> Dict:
    state = _load_state(state_path)
    if not state:
        state = {
            "created_at": _now(),
            "config":     config,
            "layouts":    {},
        }
        _save_state(state, state_path)
    return state


# ── per-layout training ───────────────────────────────────────────────────────

def run_layout(
    layout:            str,
    baselines_dir:     str,
    min_deliveries:    float,
    total_updates:     int,
    max_retries:       int,
    state:             Dict,
    state_path:        str,
    args:              argparse.Namespace,
) -> bool:
    """
    Train and validate the baseline for one layout. Returns True on success.

    Retries up to max_retries times with an offset seed on each attempt.
    On all-retry failure, logs a warning and returns False — does NOT abort
    the pipeline so other layouts can still complete.
    """
    out_path = os.path.join(baselines_dir, f"{layout}.npz")

    # ── skip if already validated ─────────────────────────────────────────────
    layout_state = state["layouts"].get(layout, {})
    if layout_state.get("status") == "complete" and os.path.exists(out_path):
        print(f"[pipeline] {layout}: already complete, skipping")
        return True

    print(f"\n[pipeline] {layout}: starting  "
          f"(min_deliveries={min_deliveries}, total_updates={total_updates})")

    shaping = {
        "pick_onion":   args.shaping_pick_onion,
        "place_onion":  args.shaping_place_onion,
        "pick_dish":    args.shaping_pick_dish,
        "deliver_soup": args.shaping_deliver_soup,
    }

    for attempt in range(1, max_retries + 1):
        attempt_seed = args.seed + (attempt - 1) * 1000
        print(f"[pipeline] {layout}: attempt {attempt}/{max_retries}  "
              f"seed={attempt_seed}")

        layout_state.update({
            "status":     "running",
            "attempt":    attempt,
            "started_at": _now(),
        })
        state["layouts"][layout] = layout_state
        _save_state(state, state_path)

        t0 = time.time()
        try:
            passed, mean_d = train_baseline(
                layout             = layout,
                out_path           = out_path,
                min_deliveries     = min_deliveries,
                n_workers          = args.n_workers,
                total_updates      = total_updates,
                pbt_interval       = args.pbt_interval,
                pbt_eval_window    = args.pbt_eval_window,
                pbt_target_pct     = args.pbt_target_pct,
                pbt_perturb        = args.pbt_perturb,
                num_steps          = args.num_steps,
                ppo_epochs         = args.ppo_epochs,
                minibatch_size     = args.minibatch_size,
                gamma              = args.gamma,
                vf_coef            = args.vf_coef,
                weight_decay       = args.weight_decay,
                hidden_sizes       = tuple(args.hidden),
                cook_time          = args.cook_time,
                save_every         = args.save_every,
                seed               = attempt_seed,
                verbose            = not args.quiet,
                shaping            = shaping,
                worker_parallelism = args.worker_parallelism,
                val_episodes       = args.val_episodes,
                val_max_steps      = args.val_max_steps,
                stuck_patience     = args.stuck_patience,
                stuck_min_delta    = args.stuck_min_delta,
                plateau_patience   = args.plateau_patience,
                plateau_min_delta  = args.plateau_min_delta,
            )
        except Exception as e:
            print(f"[pipeline] {layout}: attempt {attempt} raised exception: {e}")
            passed = False
            mean_d = 0.0

        elapsed = time.time() - t0

        if passed:
            layout_state.update({
                "status":          "complete",
                "completed_at":    _now(),
                "elapsed_s":       round(elapsed, 1),
                "mean_deliveries": round(mean_d, 3),
                "attempts":        attempt,
            })
            state["layouts"][layout] = layout_state
            _save_state(state, state_path)
            print(f"[pipeline] {layout}: PASSED "
                  f"(mean_deliveries={mean_d:.2f}, {elapsed:.0f}s)")
            return True
        else:
            print(f"[pipeline] {layout}: attempt {attempt} FAILED "
                  f"(mean_deliveries={mean_d:.2f}) — "
                  f"{'retrying' if attempt < max_retries else 'giving up'}")

    # all retries exhausted
    layout_state.update({
        "status":    "failed",
        "failed_at": _now(),
        "attempts":  max_retries,
    })
    state["layouts"][layout] = layout_state
    _save_state(state, state_path)

    print(
        f"\n[pipeline] WARNING: {layout} failed all {max_retries} attempts. "
        f"No baseline saved. The agent pipeline will fail for any agent "
        f"using this layout until the baseline is successfully trained. "
        f"Consider adjusting --total-updates, --min-deliveries, or "
        f"hyperparameters and re-running."
    )
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train and validate self-play baseline models for all 5 layouts."
    )

    ap.add_argument("--baselines-dir", type=str, default="baselines/",
                    help="Directory to save baseline .npz files (default: baselines/)")
    ap.add_argument("--min-deliveries", type=str, default=None,
                    help="Minimum mean soup deliveries/episode to pass validation. "
                         "Plain float (uniform) or JSON dict of per-layout values. "
                         "e.g. '{\"cramped_room\": 4, \"forced_coord\": 2}'. "
                         "Missing layouts use defaults.")
    ap.add_argument("--total-updates", type=str, default=None,
                    help="Total PPO updates per layout. Plain int (uniform) or "
                         "JSON dict of per-layout values. "
                         "e.g. '{\"cramped_room\": 2000, \"forced_coord\": 500}'. "
                         "Missing layouts use default of 1000.")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Max training attempts per layout before warning and "
                         "continuing (default: 3)")

    # PBT
    ap.add_argument("--n-workers",       type=int,   default=4)
    ap.add_argument("--pbt-interval",    type=int,   default=100)
    ap.add_argument("--pbt-eval-window", type=int,   default=50)
    ap.add_argument("--pbt-target-pct",  type=float, default=1.0)
    ap.add_argument("--pbt-perturb",     type=float, default=0.2)

    # PPO
    ap.add_argument("--num-steps",       type=int,   default=2048)
    ap.add_argument("--ppo-epochs",      type=int,   default=4)
    ap.add_argument("--minibatch-size",  type=int,   default=512)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--vf-coef",         type=float, default=0.5)
    ap.add_argument("--weight-decay",    type=float, default=1e-4)
    ap.add_argument("--hidden",          type=int, nargs="+", default=[256, 256])
    ap.add_argument("--cook-time",       type=int,   default=20)
    ap.add_argument("--save-every",      type=int,   default=50)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--worker-parallelism", type=str, default="sequential",
                    choices=["sequential", "parallel"])

    # reward shaping
    ap.add_argument("--shaping-pick-onion",   type=float, default=1.0)
    ap.add_argument("--shaping-place-onion",  type=float, default=5.0)
    ap.add_argument("--shaping-pick-dish",    type=float, default=2.0)
    ap.add_argument("--shaping-deliver-soup", type=float, default=5.0)

    # validation
    ap.add_argument("--val-episodes",  type=int, default=20)
    ap.add_argument("--val-max-steps", type=int, default=400)

    # early stopping (active only after min_deliveries threshold is met)
    ap.add_argument("--stuck-patience",    type=int,   default=5,
                    help="Intervals without >= stuck-min-delta improvement "
                         "before early stop (default: 5)")
    ap.add_argument("--stuck-min-delta",   type=float, default=0.25,
                    help="Minimum shaped reward improvement to reset stuck "
                         "counter (default: 0.25)")
    ap.add_argument("--plateau-patience",  type=int,   default=10,
                    help="Intervals without >= plateau-min-delta improvement "
                         "before early stop (default: 10)")
    ap.add_argument("--plateau-min-delta", type=float, default=1.0,
                    help="Minimum shaped reward improvement to reset plateau "
                         "counter (default: 1.0)")

    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.baselines_dir, exist_ok=True)

    # resolve per-layout maps (plain value or JSON dict)
    min_deliveries_map = _parse_per_layout_float(args.min_deliveries,
                                                  DEFAULT_MIN_DELIVERIES)
    total_updates_map  = _parse_per_layout_int(args.total_updates,
                                                DEFAULT_TOTAL_UPDATES)

    state_path = os.path.join(args.baselines_dir, "pipeline_state.json")
    config = {
        "total_updates":   total_updates_map,
        "n_workers":       args.n_workers,
        "seed":            args.seed,
        "min_deliveries":  min_deliveries_map,
    }
    state = _init_state(state_path, config)

    print(f"[pipeline] baselines_dir:  {args.baselines_dir}")
    print(f"[pipeline] layouts:        {ALL_LAYOUTS}")
    print(f"[pipeline] total_updates:  {total_updates_map}")
    print(f"[pipeline] min_deliveries: {min_deliveries_map}")
    print(f"[pipeline] max_retries:    {args.max_retries}")

    results: Dict[str, bool] = {}
    for layout in ALL_LAYOUTS:
        results[layout] = run_layout(
            layout         = layout,
            baselines_dir  = args.baselines_dir,
            min_deliveries = min_deliveries_map[layout],
            total_updates  = total_updates_map[layout],
            max_retries    = args.max_retries,
            state          = state,
            state_path     = state_path,
            args           = args,
        )

    # summary
    print(f"\n[pipeline] ── baseline training complete ──")
    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    for layout, ok in results.items():
        status = "OK" if ok else "FAILED"
        path   = os.path.join(args.baselines_dir, f"{layout}.npz")
        exists = os.path.exists(path)
        print(f"[pipeline]   {layout:<20} {status}  "
              f"{'(file exists)' if exists else '(NO FILE)'}")

    if n_fail > 0:
        print(
            f"\n[pipeline] WARNING: {n_fail} layout(s) failed validation. "
            f"Run the agent pipeline only after all required baselines exist."
        )
        sys.exit(1)

    print(f"\n[pipeline] all {n_ok} baselines ready in {args.baselines_dir}")


if __name__ == "__main__":
    main()