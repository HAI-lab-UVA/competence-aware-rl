"""
run_agent_pipeline.py — End-to-end controller for training agents to work
with pre-trained behavioral clones.

Reads:
  - agents.json : a list of agent configs. Each dict describes one agent
    to train (output_name, training_clones, extra_feature, layout,
    n_workers, epochs, pbt_interval — see below for defaults).
  - artifacts/partners/competence_labels.json : a nested dict {clone_name: {score_name: value, ...}}.
    Used to validate training_clones and to look up extra_feature scores.
  - clones/     : directory of <clone_name>.npz checkpoints, one per
    clone referenced in artifacts/partners/competence_labels.json / training_clones.

Each agent is trained in sequence (one at a time) by running
agent_population.py as a subprocess, writing all outputs to
<out-dir>/<output_name>/. Each agent gets its own pipeline_state.json so
a crash partway through agent N does not affect agents 0..N-1 and the
run can be resumed by simply rerunning the same command.

agents.json schema (per entry)
-------------------------------
    output_name     : str            (required)
    training_clones : list[str]      (required) — subset of clones/ to
                                        rotate through during training
    extra_features  : list[str]       (default: []) — score keys in
                                        artifacts/partners/competence_labels.json clone_1. Each key's
                                        value is appended (in order) as
                                        an extra input feature.
    layout          : str            (required) — fixed layout for the
                                        whole run (V1 or V2 name)
    n_workers       : int             (default: 4) — PBT population size
    epochs          : int             (default: 1000) — total PPO updates
    pbt_interval    : int             (default: 100) — PPO updates between
                                        PBT exploit/explore steps

Usage:
    # full run
    python run_agent_pipeline.py \
        --agents agents.json --clones-dir artifacts/partners/ --clones-scores artifacts/partners/competence_labels.json \
        --out-dir agent_runs/

    # development run (small settings, fast)
    python run_agent_pipeline.py \
        --agents agents.json --clones-dir artifacts/partners/ --clones-scores artifacts/partners/competence_labels.json \
        --out-dir agent_runs/ --dev

    # resume (just rerun the same command)
    python run_agent_pipeline.py \
        --agents agents.json --clones-dir artifacts/partners/ --clones-scores artifacts/partners/competence_labels.json \
        --out-dir agent_runs/
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

from agent_clones import load_clone_scores, validate_clones
from bc_features import LAYOUT_NAME_MAP


# ── defaults applied to agents.json entries ─────────────────────────────────

AGENT_DEFAULTS = {
    "extra_features": [],
    "n_workers":     4,
    "epochs":        1000,
    "pbt_interval":  100,
}

REQUIRED_AGENT_KEYS = ["output_name", "training_clones", "layout"]


def load_agents_config(path: str) -> List[Dict]:
    """Load agents.json, apply defaults, and do basic structural validation."""
    with open(path) as f:
        agents = json.load(f)

    if not isinstance(agents, list):
        raise ValueError("agents.json must be a JSON list of agent configs")

    resolved = []
    for i, agent in enumerate(agents):
        for key in REQUIRED_AGENT_KEYS:
            if key not in agent:
                raise ValueError(
                    f"agents.json entry {i} is missing required key '{key}'"
                )
        if not isinstance(agent["training_clones"], list) or not agent["training_clones"]:
            raise ValueError(
                f"agents.json entry {i} ('{agent['output_name']}'): "
                f"'training_clones' must be a non-empty list"
            )

        cfg = dict(AGENT_DEFAULTS)
        cfg.update(agent)
        resolved.append(cfg)

    # output_name must be unique — used as directory name
    names = [a["output_name"] for a in resolved]
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise ValueError(f"agents.json has duplicate output_name(s): {dupes}")

    return resolved


# ── state management ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(state_path: str) -> Dict:
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


def save_state(state: Dict, state_path: str):
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


def init_agent_state(state_path: str, agent_cfg: Dict) -> Dict:
    state = load_state(state_path)
    if not state:
        state = {
            "output_name": agent_cfg["output_name"],
            "config":      agent_cfg,
            "created_at":  _now(),
            "status":      "pending",
        }
        save_state(state, state_path)
    else:
        if state.get("config") != agent_cfg:
            print(f"[pipeline] WARNING: config for "
                  f"'{agent_cfg['output_name']}' differs from a previous run")
            print(f"  original: {state['config']}")
            print(f"  current:  {agent_cfg}")
    return state


# ── subprocess runner ─────────────────────────────────────────────────────────

def run_agent_subprocess(
    agent_cfg:     Dict,
    agent_idx:     int,
    clones_dir:    str,
    clones_scores: str,
    out_dir:       str,
    baselines_dir: str,
    args:          argparse.Namespace,
) -> bool:
    """
    Train one agent by running agent_population.py as a subprocess.
    Streams output to console and to a log file. Returns True on success.
    """
    agent_out_dir = os.path.join(out_dir, agent_cfg["output_name"])
    os.makedirs(agent_out_dir, exist_ok=True)
    logs_dir = os.path.join(agent_out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    state_path = os.path.join(agent_out_dir, "pipeline_state.json")
    state = init_agent_state(state_path, agent_cfg)

    if state.get("status") == "complete":
        print(f"[pipeline] agent '{agent_cfg['output_name']}': "
              f"already complete, skipping")
        return True

    population_dir = os.path.join(agent_out_dir, "population")

    # development mode uses small settings for fast testing
    effective_cfg = dict(agent_cfg)
    if args.dev:
        effective_cfg["epochs"]       = 20
        effective_cfg["pbt_interval"] = 10
        effective_cfg["n_workers"]    = 2
        num_steps = 256
    else:
        num_steps = args.num_steps

    agent_config_path = os.path.join(agent_out_dir, "agent_config.json")
    with open(agent_config_path, "w") as f:
        json.dump(effective_cfg, f, indent=2)

    cmd = [
        sys.executable, "agent_population.py",
        "--agent-config",  agent_config_path,
        "--clones-dir",    clones_dir,
        "--clones-scores", clones_scores,
        "--out-dir",       population_dir,
        "--baselines-dir", baselines_dir,
        "--agent-idx",     str(agent_idx),
        "--pbt-eval-window", str(args.pbt_eval_window),
        "--pbt-target-pct",  str(args.pbt_target_pct),
        "--pbt-perturb",     str(args.pbt_perturb),
        "--num-steps",       str(num_steps),
        "--ppo-epochs",      str(args.ppo_epochs),
        "--minibatch-size",  str(args.minibatch_size),
        "--gamma",           str(args.gamma),
        "--vf-coef",         str(args.vf_coef),
        "--weight-decay",    str(args.weight_decay),
        "--hidden"] + [str(h) for h in args.hidden] + [
        "--cook-time",       str(args.cook_time),
        "--save-every",      str(args.save_every),
        "--seed",            str(args.seed),
        "--worker-parallelism", args.worker_parallelism,
        "--shaping-pick-onion",   str(args.shaping_pick_onion),
        "--shaping-place-onion",  str(args.shaping_place_onion),
        "--shaping-pick-dish",    str(args.shaping_pick_dish),
        "--shaping-plate-soup",   str(args.shaping_plate_soup),
        "--shaping-deliver-soup", str(args.shaping_deliver_soup),
        "--eval-episodes",        str(args.eval_episodes),
        "--eval-steps",           str(args.eval_steps),
    ]
    if args.quiet:
        cmd.append("--quiet")

    log_path = os.path.join(logs_dir, "population.log")

    print(f"\n[pipeline] agent '{agent_cfg['output_name']}': starting")
    print(f"[pipeline] agent '{agent_cfg['output_name']}': log -> {log_path}")
    print(f"[pipeline] agent '{agent_cfg['output_name']}': cmd -> {' '.join(cmd)}")

    state["status"]     = "running"
    state["started_at"] = _now()
    state["log"]        = log_path
    state["cmd"]        = " ".join(cmd)
    save_state(state, state_path)

    start = time.time()
    try:
        with open(log_path, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text   = True,
                bufsize = 1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
                log_f.flush()
            process.wait()

        elapsed = time.time() - start

        if process.returncode == 0:
            state["status"]       = "complete"
            state["completed_at"] = _now()
            state["elapsed_s"]    = round(elapsed, 1)
            save_state(state, state_path)
            print(f"[pipeline] agent '{agent_cfg['output_name']}': "
                  f"complete ({elapsed:.1f}s)")
            return True
        else:
            state["status"]     = "failed"
            state["failed_at"]  = _now()
            state["returncode"] = process.returncode
            state["elapsed_s"]  = round(elapsed, 1)
            save_state(state, state_path)
            print(f"[pipeline] agent '{agent_cfg['output_name']}': "
                  f"FAILED (returncode={process.returncode})")
            print(f"[pipeline] agent '{agent_cfg['output_name']}': "
                  f"see log -> {log_path}")
            return False

    except Exception as e:
        state["status"]    = "failed"
        state["failed_at"] = _now()
        state["error"]     = str(e)
        save_state(state, state_path)
        print(f"[pipeline] agent '{agent_cfg['output_name']}': FAILED ({e})")
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train agents to work with pre-trained behavioral clones."
    )

    ap.add_argument("--agents",        type=str, required=True,
                    help="Path to agents.json")
    ap.add_argument("--clones-dir",    type=str, required=True,
                    help="Directory containing <clone_name>.npz checkpoints")
    ap.add_argument("--clones-scores", type=str, required=True,
                    help="Path to artifacts/partners/competence_labels.json")
    ap.add_argument("--out-dir",       type=str, required=True,
                    help="Output directory — each agent gets a subdirectory "
                         "named after its output_name")
    ap.add_argument("--baselines-dir", type=str, default="baselines/",
                    help="Directory containing baseline .npz checkpoints "
                         "produced by run_baseline_pipeline.py "
                         "(default: baselines/)")

    # PBT
    ap.add_argument("--pbt-eval-window", type=int,   default=50)
    ap.add_argument("--pbt-target-pct",  type=float, default=1.0)
    ap.add_argument("--pbt-perturb",     type=float, default=0.2)

    # PPO
    ap.add_argument("--num-steps",      type=int,   default=2048)
    ap.add_argument("--ppo-epochs",     type=int,   default=4)
    ap.add_argument("--minibatch-size", type=int,   default=512)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--vf-coef",        type=float, default=0.5)
    ap.add_argument("--weight-decay",   type=float, default=1e-4)
    ap.add_argument("--hidden",         type=int, nargs="+", default=[256, 256])
    ap.add_argument("--cook-time",      type=int, default=20)
    ap.add_argument("--save-every",     type=int, default=50)
    ap.add_argument("--seed",           type=int, default=0)
    ap.add_argument("--quiet",          action="store_true")

    ap.add_argument("--worker-parallelism", type=str, default="sequential",
                    choices=["sequential", "parallel"],
                    help="sequential (default): PBT workers for an agent "
                         "run one at a time — lowest memory footprint, "
                         "recommended for shared/SLURM nodes. "
                         "parallel: PBT workers run concurrently via "
                         "multiprocessing — faster wall-clock, "
                         "memory use scales with n_workers.")

    # reward shaping
    ap.add_argument("--shaping-pick-onion",   type=float, default=1.0)
    ap.add_argument("--shaping-place-onion",  type=float, default=5.0)
    ap.add_argument("--shaping-pick-dish",    type=float, default=2.0)
    ap.add_argument("--shaping-plate-soup",   type=float, default=5.0)
    ap.add_argument("--shaping-deliver-soup", type=float, default=5.0)

    # PBT-interval evaluation
    ap.add_argument("--eval-episodes", type=int, default=5,
                    help="Episodes per (clone, role) pair at each PBT interval eval")
    ap.add_argument("--eval-steps",    type=int, default=400,
                    help="Max steps per episode during PBT interval eval")

    ap.add_argument("--dev", action="store_true",
                    help="Development mode — small epochs/pbt_interval/"
                         "n_workers/num_steps overrides for fast testing")

    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[pipeline] agents:        {args.agents}")
    print(f"[pipeline] clones_dir:    {args.clones_dir}")
    print(f"[pipeline] clones_scores: {args.clones_scores}")
    print(f"[pipeline] baselines_dir: {args.baselines_dir}")
    print(f"[pipeline] out_dir:       {args.out_dir}")
    print(f"[pipeline] worker_parallelism: {args.worker_parallelism}")
    print(f"[pipeline] dev mode:      {args.dev}")

    agents = load_agents_config(args.agents)
    clone_scores = load_clone_scores(args.clones_scores)

    # validate every agent's clones up front — fail fast before training
    # anything, rather than partway through agent 3
    print(f"\n[pipeline] validating {len(agents)} agent config(s) "
          f"against artifacts/partners/competence_labels.json and {args.clones_dir}...")
    for agent in agents:
        validate_clones(
            agent["training_clones"], args.clones_dir, clone_scores,
            agent.get("extra_features", [])
        )
        # also validate validation_clones exist on disk (no scores needed)
        val_clones = agent.get("validation_clones", [])
        if val_clones:
            validate_clones(val_clones, args.clones_dir, clone_scores, [])
    print(f"[pipeline] all agent configs valid")

    # check that a baseline exists for every required layout before starting
    print(f"\n[pipeline] checking baselines in {args.baselines_dir}...")
    missing_baselines = []
    for agent in agents:
        layout_raw = agent["layout"]
        layout     = LAYOUT_NAME_MAP.get(layout_raw, layout_raw)
        path       = os.path.join(args.baselines_dir, f"{layout}.npz")
        if not os.path.exists(path):
            missing_baselines.append((agent["output_name"], layout, path))
    if missing_baselines:
        print(f"[pipeline] ERROR: missing baseline checkpoints:")
        for name, layout, path in missing_baselines:
            print(f"[pipeline]   agent '{name}' needs '{layout}' -> {path}")
        print(f"[pipeline] Run run_baseline_pipeline.py first.")
        sys.exit(1)
    print(f"[pipeline] all baselines present")

    for i, agent_cfg in enumerate(agents):
        print(f"\n[pipeline] === agent {i+1}/{len(agents)}: "
              f"'{agent_cfg['output_name']}' ===")
        ok = run_agent_subprocess(
            agent_cfg     = agent_cfg,
            agent_idx     = i,
            clones_dir    = args.clones_dir,
            clones_scores = args.clones_scores,
            out_dir       = args.out_dir,
            baselines_dir = args.baselines_dir,
            args          = args,
        )
        if not ok:
            print(f"\n[pipeline] aborting — agent "
                  f"'{agent_cfg['output_name']}' failed")
            sys.exit(1)

    print(f"\n[pipeline] complete — {len(agents)} agent(s) trained")
    for agent_cfg in agents:
        out_path = os.path.join(
            args.out_dir, agent_cfg["output_name"],
            "population", "agent_population.json"
        )
        print(f"[pipeline]   {agent_cfg['output_name']}: {out_path}")


if __name__ == "__main__":
    main()