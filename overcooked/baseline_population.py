"""
baseline_population.py — PBT orchestrator for training a baseline self-play
model for a single layout.

Training phases
---------------
Phase 1 — Pre-threshold:
    Train normally. At each PBT interval boundary, run a short self-play
    evaluation measuring mean soup deliveries per episode. Once the delivery
    count first meets min_deliveries, log the milestone and move to Phase 2.
    If total_updates is reached without meeting the threshold, return failure.

Phase 2 — Post-threshold:
    Patience counters are initialised from zero. At each subsequent PBT
    interval boundary, run a shaped-reward self-play evaluation (same shaping
    weights as training). Track the best-so-far checkpoint by shaped reward.
    Two independent early-stop conditions:
      - Stuck:   no improvement >= stuck_min_delta for stuck_patience intervals
      - Plateau: no improvement >= plateau_min_delta for plateau_patience intervals
    If either fires, stop early. Final output is always the best-so-far
    checkpoint, not the most recent one.

Workers always start from independent random inits (different seeds) to
maximise diversity in early PBT.

Usage (standalone):
    python baseline_population.py \\
        --layout cramped_room \\
        --out baselines/cramped_room.npz \\
        --min-deliveries 3
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from baseline_selfplay import BaselineSelfplayConfig, selfplay_baseline
from bc_features import FEATURE_DIM, _get_layout_stations, featurize_jaxmarl
from network import init_model, forward
from checkpoint import save_params_npz, load_params_npz

ACT_DIM = 6

# ── PBT hyperparameter space ──────────────────────────────────────────────────

HYPERPARAM_SPACE = {
    "lr":                   (1e-5, 1e-3,  "log"),
    "ent_coef":             (0.01, 0.2,   "log"),
    "clip_eps":             (0.1,  0.3,   "linear"),
    "gae_lambda":           (0.9,  0.99,  "linear"),
    "partner_update_every": (5,    20,    "linear"),
}


def sample_hyperparams(rng: np.random.Generator) -> Dict:
    params = {}
    for name, (lo, hi, scale) in HYPERPARAM_SPACE.items():
        if scale == "log":
            params[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            params[name] = float(rng.uniform(lo, hi))
        if name == "partner_update_every":
            params[name] = int(round(params[name]))
    return params


def mutate_hyperparams(
    hyperparams: Dict,
    rng:         np.random.Generator,
    perturb:     float = 0.2,
) -> Dict:
    mutated = copy.deepcopy(hyperparams)
    for name, (lo, hi, scale) in HYPERPARAM_SPACE.items():
        if scale == "log":
            factor = np.exp(rng.normal(0, perturb))
            mutated[name] = float(np.clip(mutated[name] * factor, lo, hi))
        else:
            factor = rng.uniform(1 - perturb, 1 + perturb)
            mutated[name] = float(np.clip(mutated[name] * factor, lo, hi))
        if name == "partner_update_every":
            mutated[name] = int(round(mutated[name]))
    return mutated


# ── worker state ──────────────────────────────────────────────────────────────

@dataclass
class WorkerState:
    worker_idx:     int
    hyperparams:    Dict
    checkpoint:     str
    reward_history: List[float]
    update_count:   int
    opt_state:      object = None
    partner_params: object = None
    rng_key:        object = None

    def rolling_mean_reward(self, window: int = 50) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history[-window:]))

    def to_dict(self) -> Dict:
        return {
            "worker_idx":   self.worker_idx,
            "hyperparams":  self.hyperparams,
            "checkpoint":   self.checkpoint,
            "update_count": self.update_count,
            "mean_reward":  self.rolling_mean_reward(),
        }


# ── PBT exploit / explore ─────────────────────────────────────────────────────

def pbt_step(
    workers:           List[WorkerState],
    rng:               np.random.Generator,
    hidden_sizes:      Tuple[int, ...] = (256, 256),
    target_percentile: float = 1.0,
    perturb:           float = 0.2,
    eval_window:       int   = 50,
    bottom_fraction:   float = 0.25,
) -> List[WorkerState]:
    if len(workers) < 2:
        return workers

    scored     = sorted(workers, key=lambda w: w.rolling_mean_reward(eval_window))
    n_replace  = max(1, int(len(workers) * bottom_fraction))
    target_idx = int(np.clip(
        int(target_percentile * (len(scored) - 1)), 0, len(scored) - 1
    ))
    target = scored[target_idx]

    print(
        f"[baseline_pbt] rewards: "
        f"{[f'{w.rolling_mean_reward(eval_window):.1f}' for w in scored]}  "
        f"replacing bottom {n_replace} from worker {target.worker_idx} "
        f"(target_pct={target_percentile})"
    )

    like = init_model(FEATURE_DIM, ACT_DIM, hidden_sizes=hidden_sizes)
    for i in range(n_replace):
        loser = scored[i]
        if loser is target:
            continue
        params = load_params_npz(target.checkpoint, like_params=like)
        save_params_npz(params, loser.checkpoint)
        loser.hyperparams    = mutate_hyperparams(target.hyperparams, rng, perturb)
        loser.opt_state      = None
        loser.partner_params = None
        loser.rng_key        = None
        print(f"[baseline_pbt]   worker {loser.worker_idx} replaced "
              f"from worker {target.worker_idx}")

    return workers


# ── per-worker interval runner ────────────────────────────────────────────────

def _run_worker_interval(job: Dict) -> Dict:
    """Run one PBT interval for one worker. Safe for multiprocessing."""
    cfg = BaselineSelfplayConfig(
        layout               = job["layout"],
        cook_time            = job["cook_time"],
        num_updates          = job["step"],
        num_steps            = job["num_steps"],
        ppo_epochs           = job["ppo_epochs"],
        minibatch_size       = job["minibatch_size"],
        gamma                = job["gamma"],
        gae_lambda           = job["hyperparams"]["gae_lambda"],
        clip_eps             = job["hyperparams"]["clip_eps"],
        vf_coef              = job["vf_coef"],
        ent_coef             = job["hyperparams"]["ent_coef"],
        lr                   = job["hyperparams"]["lr"],
        weight_decay         = job["weight_decay"],
        hidden_sizes         = job["hidden_sizes"],
        partner_update_every = job["hyperparams"]["partner_update_every"],
        worker_idx           = job["worker_idx"],
        out                  = job["checkpoint"],
        csv_log              = job["csv_log"],
        save_every           = job["step"],
        seed                 = job["seed"] + job["worker_idx"],
        verbose              = job["verbose"],
        shaping              = job["shaping"],
    )

    params, reward_history, opt_state, partner_params, rng_key = selfplay_baseline(
        cfg,
        checkpoint_in     = job["checkpoint"],
        opt_state         = job["opt_state"],
        partner_params_in = job["partner_params"],
        rng_key_in        = job["rng_key"],
    )

    return {
        "worker_idx":     job["worker_idx"],
        "reward_history": reward_history,
        "opt_state":      opt_state,
        "partner_params": partner_params,
        "rng_key":        rng_key,
    }


# ── interval evaluation ───────────────────────────────────────────────────────

def evaluate_interval(
    checkpoint:    str,
    layout:        str,
    hidden_sizes:  Tuple[int, ...],
    n_episodes:    int,
    max_steps:     int,
    cook_time:     int,
    shaping:       Optional[Dict],
    seed:          int,
    updates_so_far: int,
) -> Tuple[float, float]:
    """
    Run n_episodes of self-play from checkpoint, alternating seats.

    Returns (mean_deliveries, mean_shaped_reward).

    mean_deliveries  — mean soup deliveries per episode (used pre-threshold)
    mean_shaped_reward — mean total shaped reward per episode (used post-threshold)

    Both are computed in every call so we can log both regardless of phase,
    but only the relevant one drives decisions at each phase.
    """
    from jaxmarl import make
    from jaxmarl.environments.overcooked.layouts import overcooked_layouts
    from reward_shaping import compute_shaping

    like   = init_model(FEATURE_DIM, ACT_DIM, hidden_sizes=hidden_sizes)
    params = load_params_npz(checkpoint, like_params=like)

    env      = make("overcooked", layout=overcooked_layouts[layout])
    key      = jax.random.PRNGKey(seed + updates_so_far + 9999)
    key, k_r = jax.random.split(key)
    obs, state = env.reset(k_r)
    stations = _get_layout_stations(env)
    wall_map = np.array(state.wall_map)

    _shaping = shaping or {}

    all_deliveries: List[float] = []
    all_shaped:     List[float] = []

    for ep in range(n_episodes):
        learner_seat = ep % 2
        partner_seat = 1 - learner_seat
        learner_key  = f"agent_{learner_seat}"
        partner_key  = f"agent_{partner_seat}"

        key, k_r = jax.random.split(key)
        e_obs, e_state = env.reset(k_r)

        ep_deliveries   = 0.0
        ep_shaped       = 0.0
        prev_obs_arr    = None

        for _ in range(max_steps):
            key, k_l, k_p, k_step = jax.random.split(key, 4)

            feat_l = featurize_jaxmarl(
                obs        = np.array(e_obs[learner_key]),
                state      = e_state,
                player_idx = learner_seat,
                stations   = stations,
                wall_map   = wall_map,
                grid_w     = env.width,
                grid_h     = env.height,
                cook_time  = cook_time,
            )
            logits_l, _ = forward(params, jnp.asarray(feat_l))
            action_l    = int(jax.random.categorical(k_l, logits_l))

            feat_p = featurize_jaxmarl(
                obs        = np.array(e_obs[partner_key]),
                state      = e_state,
                player_idx = partner_seat,
                stations   = stations,
                wall_map   = wall_map,
                grid_w     = env.width,
                grid_h     = env.height,
                cook_time  = cook_time,
            )
            logits_p, _ = forward(params, jnp.asarray(feat_p))
            action_p    = int(jax.random.categorical(k_p, logits_p))

            actions = {
                learner_key: jnp.array(action_l, dtype=jnp.uint32),
                partner_key: jnp.array(action_p, dtype=jnp.uint32),
            }
            key, k_step = jax.random.split(key)
            curr_obs_arr = np.array(e_obs[learner_key])
            e_obs, e_state, rewards, dones, _ = env.step(
                k_step, e_state, actions
            )
            next_obs_arr = np.array(e_obs[learner_key])

            raw_r = float(rewards[learner_key])
            if raw_r > 0:
                ep_deliveries += raw_r / 20.0

            shaping_bonus = compute_shaping(
                obs_t       = next_obs_arr,
                prev_obs_t  = prev_obs_arr,
                shaping_cfg = _shaping,
                picked_from_partner_onion = False,
                picked_from_partner_dish  = False,
            )
            ep_shaped  += raw_r + shaping_bonus
            prev_obs_arr = next_obs_arr

            if dones.get("__all__", False):
                break

        all_deliveries.append(ep_deliveries)
        all_shaped.append(ep_shaped)

    return float(np.mean(all_deliveries)), float(np.mean(all_shaped))


# ── main training function ────────────────────────────────────────────────────

def train_baseline(
    layout:             str,
    out_path:           str,
    min_deliveries:     float,
    n_workers:          int   = 4,
    total_updates:      int   = 1000,
    pbt_interval:       int   = 100,
    pbt_eval_window:    int   = 50,
    pbt_target_pct:     float = 1.0,
    pbt_perturb:        float = 0.2,
    num_steps:          int   = 2048,
    ppo_epochs:         int   = 4,
    minibatch_size:     int   = 512,
    gamma:              float = 0.99,
    vf_coef:            float = 0.5,
    weight_decay:       float = 1e-4,
    hidden_sizes:       Tuple = (256, 256),
    cook_time:          int   = 20,
    save_every:         int   = 50,
    seed:               int   = 0,
    verbose:            bool  = True,
    shaping:            dict  = None,
    worker_parallelism: str   = "sequential",
    val_episodes:       int   = 20,
    val_max_steps:      int   = 400,
    stuck_patience:     int   = 5,
    stuck_min_delta:    float = 0.25,
    plateau_patience:   int   = 10,
    plateau_min_delta:  float = 1.0,
) -> Tuple[bool, float]:
    """
    Train a baseline model for one layout via PBT self-play.

    See module docstring for the two-phase training logic.

    Returns (validation_passed: bool, mean_deliveries: float).
    The caller (run_baseline_pipeline.py) is responsible for retries on failure.
    """
    if shaping is None:
        shaping = {
            "pick_onion":   1.0,
            "place_onion":  5.0,
            "pick_dish":    2.0,
            "deliver_soup": 5.0,
        }

    out_dir       = os.path.dirname(out_path) or "."
    pbt_dir       = os.path.join(out_dir, f"{layout}_pbt")
    snapshots_dir = os.path.join(out_dir, f"{layout}_snapshots")
    best_val_path = os.path.join(out_dir, f"{layout}_best.npz")
    os.makedirs(pbt_dir,       exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── initialise workers from independent random inits ─────────────────────
    workers: List[WorkerState] = []
    for w in range(n_workers):
        worker_ckpt = os.path.join(pbt_dir, f"worker_{w:02d}.npz")
        if not os.path.exists(worker_ckpt):
            init_key = jax.random.fold_in(jax.random.PRNGKey(seed), w)
            params   = init_model(
                FEATURE_DIM, ACT_DIM,
                hidden_sizes=hidden_sizes,
                key=init_key,
            )
            save_params_npz(params, worker_ckpt)

        workers.append(WorkerState(
            worker_idx     = w,
            hyperparams    = sample_hyperparams(rng),
            checkpoint     = worker_ckpt,
            reward_history = [],
            update_count   = 0,
        ))

    # epoch-0 snapshots
    like = init_model(FEATURE_DIM, ACT_DIM, hidden_sizes=hidden_sizes)
    for w in workers:
        snap0 = os.path.join(
            snapshots_dir, f"worker_{w.worker_idx:02d}_at_update_0000.npz"
        )
        if not os.path.exists(snap0):
            params = load_params_npz(w.checkpoint, like_params=like)
            save_params_npz(params, snap0)

    print(f"[baseline {layout}] {n_workers} workers initialised")

    # ── phase tracking ────────────────────────────────────────────────────────
    threshold_met    = False   # have we passed min_deliveries yet?
    best_val_reward  = -float("inf")   # best shaped reward seen post-threshold
    stuck_counter    = 0
    plateau_counter  = 0
    early_stop       = False

    # ── PBT loop ──────────────────────────────────────────────────────────────
    updates_so_far = 0

    while updates_so_far < total_updates and not early_stop:
        step = min(
            pbt_interval - (updates_so_far % pbt_interval),
            total_updates - updates_so_far,
        )

        print(
            f"[baseline {layout}] "
            f"updates {updates_so_far}→{updates_so_far + step}  "
            f"phase={'post-threshold' if threshold_met else 'pre-threshold'}"
        )

        jobs = []
        for worker in workers:
            csv_log = os.path.join(pbt_dir, f"worker_{worker.worker_idx:02d}.csv")
            jobs.append({
                "worker_idx":     worker.worker_idx,
                "hyperparams":    worker.hyperparams,
                "checkpoint":     worker.checkpoint,
                "opt_state":      worker.opt_state,
                "partner_params": worker.partner_params,
                "rng_key":        worker.rng_key,
                "layout":         layout,
                "cook_time":      cook_time,
                "step":           step,
                "num_steps":      num_steps,
                "ppo_epochs":     ppo_epochs,
                "minibatch_size": minibatch_size,
                "gamma":          gamma,
                "vf_coef":        vf_coef,
                "weight_decay":   weight_decay,
                "hidden_sizes":   hidden_sizes,
                "save_every":     save_every,
                "seed":           seed,
                "verbose":        verbose,
                "shaping":        shaping,
                "csv_log":        csv_log,
            })

        if worker_parallelism == "parallel" and n_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(_run_worker_interval, jobs)
        else:
            results = [_run_worker_interval(job) for job in jobs]

        results_by_idx = {r["worker_idx"]: r for r in results}
        for worker in workers:
            r = results_by_idx[worker.worker_idx]
            worker.reward_history.extend(r["reward_history"])
            worker.update_count   += step
            worker.opt_state       = r["opt_state"]
            worker.partner_params  = r["partner_params"]
            worker.rng_key         = r["rng_key"]

        updates_so_far += step
        at_pbt_boundary = (updates_so_far % pbt_interval == 0)

        if at_pbt_boundary and updates_so_far < total_updates:
            workers = pbt_step(
                workers           = workers,
                rng               = rng,
                hidden_sizes      = hidden_sizes,
                target_percentile = pbt_target_pct,
                perturb           = pbt_perturb,
                eval_window       = pbt_eval_window,
            )

        if at_pbt_boundary:
            # save snapshots
            for w in workers:
                snap = os.path.join(
                    snapshots_dir,
                    f"worker_{w.worker_idx:02d}_at_update_{updates_so_far:04d}.npz",
                )
                params = load_params_npz(w.checkpoint, like_params=like)
                save_params_npz(params, snap)

            best = max(workers,
                       key=lambda w: w.rolling_mean_reward(pbt_eval_window))
            print(
                f"[baseline {layout}] snapshots saved at update {updates_so_far}  "
                f"best worker {best.worker_idx}  "
                f"reward={best.rolling_mean_reward(pbt_eval_window):.1f}"
            )

            # ── interval evaluation ───────────────────────────────────────────
            mean_d, mean_shaped = evaluate_interval(
                checkpoint     = best.checkpoint,
                layout         = layout,
                hidden_sizes   = hidden_sizes,
                n_episodes     = val_episodes,
                max_steps      = val_max_steps,
                cook_time      = cook_time,
                shaping        = shaping,
                seed           = seed,
                updates_so_far = updates_so_far,
            )

            print(
                f"[baseline {layout}] interval eval @ update {updates_so_far}  "
                f"mean_deliveries={mean_d:.2f}  "
                f"mean_shaped_reward={mean_shaped:.2f}"
            )

            # ── phase 1: check if threshold is first met ──────────────────────
            if not threshold_met:
                if mean_d >= min_deliveries:
                    threshold_met = True
                    print(
                        f"\n[baseline {layout}] *** THRESHOLD MET at update "
                        f"{updates_so_far} ***\n"
                        f"  mean_deliveries={mean_d:.2f} >= {min_deliveries}\n"
                        f"  Activating early-stop patience counters.\n"
                        f"  Switching evaluation signal to shaped reward.\n"
                        f"  stuck_patience={stuck_patience}  "
                        f"stuck_min_delta={stuck_min_delta}\n"
                        f"  plateau_patience={plateau_patience}  "
                        f"plateau_min_delta={plateau_min_delta}\n"
                    )
                    # save initial best-so-far from the moment threshold is met
                    params = load_params_npz(best.checkpoint, like_params=like)
                    save_params_npz(params, best_val_path)
                    best_val_reward = mean_shaped
                    print(
                        f"[baseline {layout}] initial best-val checkpoint saved "
                        f"(shaped_reward={best_val_reward:.2f})"
                    )
                else:
                    print(
                        f"[baseline {layout}] threshold not yet met "
                        f"({mean_d:.2f} < {min_deliveries})"
                    )

            # ── phase 2: post-threshold patience tracking ─────────────────────
            elif threshold_met:
                improvement = mean_shaped - best_val_reward

                # update best-so-far if we improved
                if mean_shaped > best_val_reward:
                    best_val_reward = mean_shaped
                    params = load_params_npz(best.checkpoint, like_params=like)
                    save_params_npz(params, best_val_path)
                    print(
                        f"[baseline {layout}] new best-val checkpoint saved "
                        f"(shaped_reward={best_val_reward:.2f})"
                    )

                # update patience counters
                if improvement >= stuck_min_delta:
                    stuck_counter = 0
                else:
                    stuck_counter += 1

                if improvement >= plateau_min_delta:
                    plateau_counter = 0
                else:
                    plateau_counter += 1

                print(
                    f"[baseline {layout}] post-threshold  "
                    f"shaped_reward={mean_shaped:.2f}  "
                    f"best={best_val_reward:.2f}  "
                    f"improvement={improvement:.2f}  "
                    f"stuck={stuck_counter}/{stuck_patience}  "
                    f"plateau={plateau_counter}/{plateau_patience}"
                )

                if stuck_counter >= stuck_patience:
                    print(
                        f"\n[baseline {layout}] EARLY STOP — "
                        f"no improvement >= {stuck_min_delta} for "
                        f"{stuck_patience} consecutive intervals (stuck)"
                    )
                    early_stop = True

                elif plateau_counter >= plateau_patience:
                    print(
                        f"\n[baseline {layout}] EARLY STOP — "
                        f"no improvement >= {plateau_min_delta} for "
                        f"{plateau_patience} consecutive intervals (plateau)"
                    )
                    early_stop = True

    # ── select output checkpoint ──────────────────────────────────────────────
    if not threshold_met:
        # never reached the delivery threshold — training failed
        print(
            f"[baseline {layout}] training complete but threshold NEVER MET "
            f"(min_deliveries={min_deliveries}) — validation failed"
        )
        return False, 0.0

    # use best-so-far (saved during post-threshold phase), not most recent
    print(
        f"\n[baseline {layout}] training complete  "
        f"{'(early stop)' if early_stop else '(full run)'}  "
        f"best_shaped_reward={best_val_reward:.2f}"
    )

    # final delivery validation on the best-val checkpoint
    mean_d, _ = evaluate_interval(
        checkpoint     = best_val_path,
        layout         = layout,
        hidden_sizes   = hidden_sizes,
        n_episodes     = val_episodes,
        max_steps      = val_max_steps,
        cook_time      = cook_time,
        shaping        = shaping,
        seed           = seed + 1,
        updates_so_far = updates_so_far + 1,
    )

    passed = mean_d >= min_deliveries
    print(
        f"[baseline {layout}] final validation  "
        f"mean_deliveries={mean_d:.2f}  "
        f"threshold={min_deliveries}  "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    if passed:
        # atomic replace — only write the output file on confirmed pass
        import shutil
        shutil.copy2(best_val_path, out_path)
        print(f"[baseline {layout}] saved -> {out_path}")
    else:
        print(f"[baseline {layout}] not saved")

    return passed, mean_d


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="PBT self-play baseline training for one layout."
    )
    ap.add_argument("--layout",          type=str,   required=True)
    ap.add_argument("--out",             type=str,   required=True,
                    help="Output path for the validated baseline .npz")
    ap.add_argument("--min-deliveries",  type=float, required=True,
                    help="Minimum mean soup deliveries per episode to pass "
                         "validation and activate early stopping")

    ap.add_argument("--n-workers",         type=int,   default=4)
    ap.add_argument("--total-updates",     type=int,   default=1000)
    ap.add_argument("--pbt-interval",      type=int,   default=100)
    ap.add_argument("--pbt-eval-window",   type=int,   default=50)
    ap.add_argument("--pbt-target-pct",    type=float, default=1.0)
    ap.add_argument("--pbt-perturb",       type=float, default=0.2)
    ap.add_argument("--num-steps",         type=int,   default=2048)
    ap.add_argument("--ppo-epochs",        type=int,   default=4)
    ap.add_argument("--minibatch-size",    type=int,   default=512)
    ap.add_argument("--gamma",             type=float, default=0.99)
    ap.add_argument("--vf-coef",           type=float, default=0.5)
    ap.add_argument("--weight-decay",      type=float, default=1e-4)
    ap.add_argument("--hidden",            type=int, nargs="+", default=[256, 256])
    ap.add_argument("--cook-time",         type=int,   default=20)
    ap.add_argument("--save-every",        type=int,   default=50)
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--worker-parallelism", type=str,  default="sequential",
                    choices=["sequential", "parallel"])
    ap.add_argument("--val-episodes",      type=int,   default=20)
    ap.add_argument("--val-max-steps",     type=int,   default=400)

    # early stopping (active only after min_deliveries threshold is met)
    ap.add_argument("--stuck-patience",    type=int,   default=5,
                    help="Intervals without >= stuck-min-delta improvement "
                         "before early stop (default: 5)")
    ap.add_argument("--stuck-min-delta",   type=float, default=0.25,
                    help="Minimum shaped reward improvement to reset stuck "
                         "counter (default: 0.25, ~1/4 delivery)")
    ap.add_argument("--plateau-patience",  type=int,   default=10,
                    help="Intervals without >= plateau-min-delta improvement "
                         "before early stop (default: 10)")
    ap.add_argument("--plateau-min-delta", type=float, default=1.0,
                    help="Minimum shaped reward improvement to reset plateau "
                         "counter (default: 1.0, ~1/20 delivery)")

    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    passed, mean_d = train_baseline(
        layout             = args.layout,
        out_path           = args.out,
        min_deliveries     = args.min_deliveries,
        n_workers          = args.n_workers,
        total_updates      = args.total_updates,
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
        seed               = args.seed,
        verbose            = not args.quiet,
        worker_parallelism = args.worker_parallelism,
        val_episodes       = args.val_episodes,
        val_max_steps      = args.val_max_steps,
        stuck_patience     = args.stuck_patience,
        stuck_min_delta    = args.stuck_min_delta,
        plateau_patience   = args.plateau_patience,
        plateau_min_delta  = args.plateau_min_delta,
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()