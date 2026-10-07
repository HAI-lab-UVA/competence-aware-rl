"""
agent_population.py — PBT orchestrator for training ONE agent against a
population of frozen behavioral clones.

For a single agent config (one entry from agents.json):
  1. Load the pre-trained baseline checkpoint for this layout from
     baselines_dir/<layout>.npz.  The baseline was trained via self-play
     with only base features (FEATURE_DIM).  If the agent uses extra
     features, the first-layer input weights are zero-extended to
     FEATURE_DIM + len(extra_features) — existing base-feature columns are
     preserved exactly, new columns start at zero so the policy initially
     ignores them and learns to use them gradually.
  2. Expand training_clones into a pool of (clone_name, role) pairs.
     String entries expand to both clone_0 and clone_1; list entries use
     the specified role only.
  3. Spawn n_workers PBT workers, each initialised from the baseline.
  4. Each PBT interval, generate a per-update shuffled schedule for every
     worker (randomly sampling from the pool, one pair per PPO update).
     The partner clone AND its seat can therefore change every update.
  5. At each PBT boundary, exploit/explore is applied across workers AND
     the best worker is evaluated against validation_clones (both roles).
     Early stopping fires when either:
       - No improvement >= stuck_min_delta for stuck_patience intervals
       - No improvement >= plateau_min_delta for plateau_patience intervals
     On early stop the best-validation checkpoint so far is saved as final.
  6. After `epochs` total PPO updates (or early stop), the best worker's
     checkpoint is saved as the agent's final checkpoint.

Workers can run sequentially (default — minimises memory footprint, one
JAX runtime alive at a time) or in parallel via multiprocessing.

Usage:
    python agent_population.py \
        --agent-config agent_cfg.json \
        --clones-dir artifacts/partners/ \
        --clones-scores artifacts/partners/competence_labels.json \
        --baselines-dir artifacts/baselines/ \
        --out-dir population_out/agent_0/
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

from agent_finetune import AgentFinetuneConfig, finetune_vs_clone
from agent_clones import (
    load_clone_scores, validate_clones, clone_checkpoint_path,
    clone_partner_scores, expand_training_clones,
)
from bc_features import FEATURE_DIM, LAYOUT_NAME_MAP
from network import init_model
from checkpoint import save_params_npz, load_params_npz

ACT_DIM = 6

# ── baseline loading ──────────────────────────────────────────────────────────

def load_baseline_params(
    baselines_dir: str,
    layout:        str,
    obs_dim:       int,
    hidden_sizes:  Tuple[int, ...],
    seed:          int = 0,
):
    """
    Load the pre-trained self-play baseline for this layout and zero-extend
    the first-layer input weights to obs_dim if obs_dim > FEATURE_DIM.

    The baseline was trained with obs_dim == FEATURE_DIM (base features only).
    When extra features are used, the first layer's weight matrix W has shape
    (hidden_0, FEATURE_DIM) and must become (hidden_0, obs_dim).  We pad the
    right side with zeros so the extra-feature columns contribute nothing on
    the first forward pass — the model starts behaving exactly like the
    baseline and learns to use the new features gradually.

    All other layers (hidden and output) are copied unchanged.

    Raises FileNotFoundError if the baseline does not exist.
    """
    baseline_path = os.path.join(baselines_dir, f"{layout}.npz")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline checkpoint not found for layout '{layout}' at "
            f"{baseline_path}. Run run_baseline_pipeline.py first."
        )

    # load with base obs_dim structure
    base_like = init_model(
        FEATURE_DIM, ACT_DIM,
        hidden_sizes=hidden_sizes,
        key=jax.random.PRNGKey(seed),
    )
    base_params = load_params_npz(baseline_path, like_params=base_like)

    if obs_dim == FEATURE_DIM:
        # no extension needed
        return base_params

    # zero-extend only the first layer's W: (hidden_0, FEATURE_DIM)
    # → (hidden_0, obs_dim) by appending zero columns on the right
    n_extra      = obs_dim - FEATURE_DIM
    old_W, old_b = base_params["layers"][0]
    hidden_0     = old_W.shape[0]
    zero_cols    = jnp.zeros((hidden_0, n_extra), dtype=old_W.dtype)
    new_W        = jnp.concatenate([old_W, zero_cols], axis=1)

    new_layers        = list(base_params["layers"])
    new_layers[0]     = (new_W, old_b)
    extended_params   = dict(base_params)
    extended_params["layers"] = new_layers

    return extended_params

# ── PBT hyperparameter space ────────────────────────────────────────────────
# Same search space as bc_population.py, minus partner_update_every —
# that hyperparameter has no meaning against a frozen clone partner.

HYPERPARAM_SPACE = {
    "lr":         (1e-5, 1e-3, "log"),
    "ent_coef":   (0.01, 0.2,  "log"),
    "clip_eps":   (0.1,  0.3,  "linear"),
    "gae_lambda": (0.9,  0.99, "linear"),
}


def sample_hyperparams(rng: np.random.Generator) -> Dict:
    """Sample a random hyperparameter configuration from the search space."""
    params = {}
    for name, (lo, hi, scale) in HYPERPARAM_SPACE.items():
        if scale == "log":
            params[name] = float(np.exp(
                rng.uniform(np.log(lo), np.log(hi))
            ))
        else:
            params[name] = float(rng.uniform(lo, hi))
    return params


def mutate_hyperparams(
    hyperparams: Dict,
    rng:         np.random.Generator,
    perturb:     float = 0.2,
) -> Dict:
    """Perturb hyperparameters by a random factor, clipped to search space."""
    mutated = copy.deepcopy(hyperparams)
    for name, (lo, hi, scale) in HYPERPARAM_SPACE.items():
        if scale == "log":
            factor = np.exp(rng.normal(0, perturb))
            mutated[name] = float(np.clip(mutated[name] * factor, lo, hi))
        else:
            factor = rng.uniform(1 - perturb, 1 + perturb)
            mutated[name] = float(np.clip(mutated[name] * factor, lo, hi))
    return mutated


# ── PBT worker state ──────────────────────────────────────────────────────────

@dataclass
class WorkerState:
    """
    Tracks the state of one PBT worker during fine-tuning.

    agent_idx      : which agent (from agents.json) this worker belongs to
    worker_idx     : which worker within this agent's PBT population
    hyperparams    : current hyperparameter configuration
    checkpoint     : path to this worker's latest checkpoint
    reward_history : rolling history of per-update rewards
    update_count   : how many PPO updates have been completed
    opt_state      : Adam state, threaded across finetune calls
    rng_key        : JAX RNG key, threaded across finetune calls
    """
    agent_idx:      int
    worker_idx:     int
    hyperparams:    Dict
    checkpoint:     str
    reward_history: List[float]
    update_count:   int
    opt_state:      object = None
    rng_key:        object = None

    def rolling_mean_reward(self, window: int = 50) -> float:
        if not self.reward_history:
            return 0.0
        recent = self.reward_history[-window:]
        return float(np.mean(recent))

    def to_dict(self) -> Dict:
        return {
            "agent_idx":    self.agent_idx,
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
    obs_dim:           int,
    hidden_sizes:      Tuple[int, ...] = (256, 256),
    target_percentile: float = 1.0,
    perturb:           float = 0.2,
    eval_window:       int   = 50,
    bottom_fraction:   float = 0.25,
) -> List[WorkerState]:
    """
    One PBT exploit/explore step across this agent's workers.

    1. Rank workers by rolling mean reward.
    2. Bottom `bottom_fraction` workers copy weights from the target
       percentile worker.
    3. Copied workers mutate their hyperparameters.
    4. Top workers keep their weights and hyperparameters unchanged.
    """
    if len(workers) < 2:
        return workers

    scored = sorted(workers, key=lambda w: w.rolling_mean_reward(eval_window))

    n_replace  = max(1, int(len(workers) * bottom_fraction))
    target_idx = int(target_percentile * (len(scored) - 1))
    target_idx = int(np.clip(target_idx, 0, len(scored) - 1))
    target     = scored[target_idx]

    print(f"[pbt] agent {workers[0].agent_idx}  "
          f"rewards: {[f'{w.rolling_mean_reward(eval_window):.1f}' for w in scored]}  "
          f"replacing bottom {n_replace} workers from "
          f"worker {target.worker_idx} "
          f"(target_percentile={target_percentile})")

    for i in range(n_replace):
        loser = scored[i]
        if loser is target:
            continue

        # copy target checkpoint to loser's checkpoint path
        params = load_params_npz(
            target.checkpoint,
            like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
        )
        save_params_npz(params, loser.checkpoint)

        # explore: mutate hyperparameters
        loser.hyperparams = mutate_hyperparams(
            target.hyperparams, rng, perturb=perturb
        )

        # stale training state — reinitialised on next finetune call
        loser.opt_state = None
        loser.rng_key   = None

        print(f"[pbt]   worker {loser.worker_idx} replaced "
              f"with mutated hyperparams from worker {target.worker_idx}")

    return workers


# ── PBT-interval evaluation ───────────────────────────────────────────────────

def evaluate_at_interval(
    best_worker:    "WorkerState",
    clone_pool:     List[Tuple[str, str]],
    clone_scores:   Dict,
    clones_dir:     str,
    extra_features: List[str],
    layout:         str,
    cook_time:      int,
    hidden_sizes:   Tuple,
    obs_dim:        int,
    updates_so_far: int,
    eval_episodes:  int,
    eval_steps:     int,
    eval_csv:       str,
    verbose:        bool = True,
) -> List[Dict]:
    """
    Run short evaluation rollouts of the best worker's current checkpoint
    against every unique (clone_name, clone_role) pair in the clone pool.
    Logs mean reward per pair to eval_csv.

    Runs eval_episodes rollouts per pair, each of length eval_steps steps.
    Returns a list of result dicts (one per pair).
    """
    import csv as _csv
    from jaxmarl import make
    from jaxmarl.environments.overcooked.layouts import overcooked_layouts
    from bc_features import _get_layout_stations, featurize_jaxmarl
    from network import forward
    from checkpoint import load_params_npz
    from agent_clones import (
        ClonePartner, clone_checkpoint_path,
        learner_player_idx, clone_player_idx, clone_partner_scores,
    )

    # load best worker params
    like = init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
    params = load_params_npz(best_worker.checkpoint, like_params=like)

    # load each unique clone once
    unique_clones = {name for name, _ in clone_pool}
    partners: Dict[str, ClonePartner] = {}
    for name in unique_clones:
        partners[name] = ClonePartner(
            name         = name,
            checkpoint   = clone_checkpoint_path(clones_dir, name),
            hidden_sizes = hidden_sizes,
            seed         = 0,
        )

    env      = make("overcooked", layout=overcooked_layouts[layout])
    key      = jax.random.PRNGKey(updates_so_far)
    key, k_r = jax.random.split(key)
    obs, state = env.reset(k_r)
    stations = _get_layout_stations(env)
    wall_map = np.array(state.wall_map)

    # deduplicate pool while preserving order
    seen   = set()
    unique_pairs = []
    for pair in clone_pool:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    results = []
    write_header = not os.path.exists(eval_csv)
    os.makedirs(os.path.dirname(eval_csv) or ".", exist_ok=True)

    with open(eval_csv, "a", newline="") as f:
        writer = _csv.writer(f)
        if write_header:
            writer.writerow([
                "updates_so_far", "clone_name", "clone_role",
                "learner_seat", "mean_reward", "episodes",
            ])

        for clone_name, clone_role in unique_pairs:
            l_idx = learner_player_idx(clone_role)
            c_idx = clone_player_idx(clone_role)

            extra_vals = clone_partner_scores(
                clone_scores, clone_name, clone_role, extra_features
            )

            partner_fn = partners[clone_name].make_partner_fn(
                stations   = stations,
                wall_map   = wall_map,
                grid_w     = env.width,
                grid_h     = env.height,
                player_idx = c_idx,
                cook_time  = cook_time,
            )

            extra_feat_fn = None
            if extra_vals:
                _v = list(extra_vals)
                def extra_feat_fn(_vals=_v):
                    return _vals

            episode_rewards = []
            for ep in range(eval_episodes):
                key, k_r = jax.random.split(key)
                obs, state = env.reset(k_r)
                ep_reward  = 0.0
                prev_action = 4

                for _ in range(eval_steps):
                    # featurize learner obs
                    agent_key = f"agent_{l_idx}"
                    feat = featurize_jaxmarl(
                        obs        = np.array(obs[agent_key]),
                        state      = state,
                        player_idx = l_idx,
                        stations   = stations,
                        wall_map   = wall_map,
                        grid_w     = env.width,
                        grid_h     = env.height,
                        cook_time  = cook_time,
                    )
                    if extra_feat_fn is not None:
                        feat = np.concatenate(
                            [feat, np.array(extra_feat_fn(), dtype=np.float32)]
                        )

                    key, k_a, k_p = jax.random.split(key, 3)
                    import jax.numpy as jnp
                    logits, _ = forward(params, jnp.asarray(feat))
                    action = int(jax.random.categorical(k_a, logits))

                    # partner action
                    partner_agent_key = f"agent_{c_idx}"
                    p_action = int(partner_fn(
                        obs[partner_agent_key], state, k_p
                    ))

                    actions = {
                        f"agent_{l_idx}": action,
                        f"agent_{c_idx}": p_action,
                    }
                    key, k_step = jax.random.split(key)
                    obs, state, rewards, dones, _ = env.step(k_step, state, actions)
                    ep_reward += float(rewards[f"agent_{l_idx}"])

                    if dones.get("__all__", False):
                        break

                episode_rewards.append(ep_reward)

            mean_reward = float(np.mean(episode_rewards))
            row = {
                "updates_so_far": updates_so_far,
                "clone_name":     clone_name,
                "clone_role":     clone_role,
                "learner_seat":   l_idx,
                "mean_reward":    mean_reward,
                "episodes":       eval_episodes,
            }
            results.append(row)
            writer.writerow([
                updates_so_far, clone_name, clone_role,
                l_idx, f"{mean_reward:.3f}", eval_episodes,
            ])
            f.flush()

            if verbose:
                print(f"[eval] update={updates_so_far}  "
                      f"{clone_name}({clone_role})  "
                      f"seat={l_idx}  "
                      f"reward={mean_reward:.2f}")

    return results




def _run_worker_interval(job: Dict) -> Dict:
    """
    Run one PBT interval (`step` PPO updates) for one worker. The partner
    clone and its role are determined per-update from job["update_schedule"],
    a pre-generated list of (clone_name, role) pairs of length `step`.

    Designed to be called either directly (sequential mode) or via
    multiprocessing.Pool.map (parallel mode).

    Returns a dict with the updated reward_history, opt_state, and rng_key.
    """
    cfg = AgentFinetuneConfig(
        layout                         = job["layout"],
        cook_time                      = job["cook_time"],
        num_updates                    = job["step"],
        num_steps                      = job["num_steps"],
        ppo_epochs                     = job["ppo_epochs"],
        minibatch_size                 = job["minibatch_size"],
        gamma                          = job["gamma"],
        gae_lambda                     = job["hyperparams"]["gae_lambda"],
        clip_eps                       = job["hyperparams"]["clip_eps"],
        vf_coef                        = job["vf_coef"],
        ent_coef                       = job["hyperparams"]["ent_coef"],
        lr                             = job["hyperparams"]["lr"],
        weight_decay                   = job["weight_decay"],
        hidden_sizes                   = job["hidden_sizes"],
        agent_idx                      = job["agent_idx"],
        worker_idx                     = job["worker_idx"],
        out                            = job["checkpoint"],
        csv_log                        = job["csv_log"],
        save_every                     = job["step"],
        seed                           = job["seed"] + job["agent_idx"] + job["worker_idx"],
        verbose                        = job["verbose"],
        shaping                        = job["shaping"],
        update_schedule                = job["update_schedule"],
        extra_feature_values_schedule  = job["extra_feature_values_schedule"],
        clones_dir                     = job["clones_dir"],
    )

    params, reward_history, opt_state, rng_key = finetune_vs_clone(
        cfg,
        checkpoint_in = job["checkpoint"],
        opt_state     = job["opt_state"],
        rng_key_in    = job["rng_key"],
    )

    return {
        "worker_idx":     job["worker_idx"],
        "reward_history": reward_history,
        "opt_state":      opt_state,
        "rng_key":        rng_key,
    }


# ── per-agent training ─────────────────────────────────────────────────────────

def train_agent(
    agent_cfg:          Dict,
    clones_dir:         str,
    clone_scores:       Dict[str, Dict[str, float]],
    out_dir:            str,
    baselines_dir:      str   = "baselines/",
    agent_idx:          int   = 0,
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
    eval_episodes:      int   = 5,
    eval_steps:         int   = 400,
) -> Dict:
    """
    Train a single agent (from one agents.json entry) via PBT against its
    training_clones.

    agent_cfg keys (defaults applied by run_agent_pipeline.py before
    calling this):
        output_name    : str
        training_clones: list[str]  — clone names to rotate through
        extra_features : list[str] — score keys in artifacts/partners/competence_labels.json
        layout         : str        — fixed layout for the whole run
        n_workers      : int        — PBT population size
        epochs         : int        — total PPO updates (== num_updates)
        pbt_interval   : int        — PPO updates between PBT steps

    Returns a dict summarising the agent's final state, also written to
    agent_population.json under out_dir.
    """
    if shaping is None:
        shaping = {
            "pick_onion":   1.0,
            "place_onion":  5.0,
            "pick_dish":    2.0,
            "deliver_soup": 5.0,
        }

    output_name       = agent_cfg["output_name"]
    training_clones   = agent_cfg["training_clones"]
    validation_clones = agent_cfg.get("validation_clones", [])
    extra_features    = agent_cfg.get("extra_features", [])
    layout_raw        = agent_cfg["layout"]
    layout            = LAYOUT_NAME_MAP.get(layout_raw, layout_raw)
    n_workers         = agent_cfg["n_workers"]
    total_updates     = agent_cfg["epochs"]
    pbt_interval      = agent_cfg["pbt_interval"]

    # early-stopping thresholds (with defaults)
    stuck_patience    = agent_cfg.get("stuck_patience",    5)
    stuck_min_delta   = agent_cfg.get("stuck_min_delta",   5.0)
    plateau_patience  = agent_cfg.get("plateau_patience",  10)
    plateau_min_delta = agent_cfg.get("plateau_min_delta", 20.0)

    # validate training clones up front
    validate_clones(training_clones, clones_dir, clone_scores, extra_features)

    # validate validation clones (no extra_features needed — scores not used)
    if validation_clones:
        validate_clones(validation_clones, clones_dir, clone_scores, [])

    clone_pool = expand_training_clones(training_clones)
    # validation pool always uses both roles
    val_pool   = expand_training_clones(validation_clones) if validation_clones else []

    obs_dim = FEATURE_DIM + len(extra_features)

    os.makedirs(out_dir, exist_ok=True)
    pbt_dir       = os.path.join(out_dir, "pbt")
    snapshots_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(pbt_dir,       exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)

    result_path      = os.path.join(out_dir, "agent_population.json")
    final_checkpoint = os.path.join(out_dir, "final.npz")
    best_val_checkpoint = os.path.join(out_dir, "best_val.npz")

    # resume if already complete
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        if existing.get("status") == "complete":
            print(f"[agent {agent_idx}] {output_name}: already complete, skipping")
            return existing

    rng = np.random.default_rng(seed + agent_idx)

    # ── load baseline and initialise workers ─────────────────────────────────
    print(f"[agent {agent_idx}] {output_name}: loading baseline for '{layout}'")
    baseline_params = load_baseline_params(
        baselines_dir = baselines_dir,
        layout        = layout,
        obs_dim       = obs_dim,
        hidden_sizes  = hidden_sizes,
        seed          = seed,
    )
    print(f"[agent {agent_idx}] {output_name}: baseline loaded "
          f"(obs_dim={obs_dim}, base={FEATURE_DIM}, "
          f"extra={obs_dim - FEATURE_DIM})")

    workers: List[WorkerState] = []
    for w in range(n_workers):
        worker_ckpt = os.path.join(pbt_dir, f"worker_{w:02d}.npz")
        if not os.path.exists(worker_ckpt):
            save_params_npz(baseline_params, worker_ckpt)

        workers.append(WorkerState(
            agent_idx      = agent_idx,
            worker_idx     = w,
            hyperparams    = sample_hyperparams(rng),
            checkpoint     = worker_ckpt,
            reward_history = [],
            update_count   = 0,
        ))

    # epoch-0 snapshots
    for w_idx in range(n_workers):
        snap0 = os.path.join(snapshots_dir, f"worker_{w_idx:02d}_at_update_0000.npz")
        if not os.path.exists(snap0):
            params0 = load_params_npz(
                workers[w_idx].checkpoint,
                like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
            )
            save_params_npz(params0, snap0)
    print(f"[agent {agent_idx}] {output_name}: snapshots saved — "
          f"all {n_workers} workers at update 0")

    # ── early-stopping state ─────────────────────────────────────────────────
    # best_val_reward: best mean validation reward seen so far
    # stuck_counter:   intervals without >= stuck_min_delta improvement
    # plateau_counter: intervals without >= plateau_min_delta improvement
    best_val_reward   = -float("inf")
    stuck_counter     = 0
    plateau_counter   = 0
    early_stop        = False

    # ── PBT fine-tuning loop ─────────────────────────────────────────────────
    updates_so_far = 0
    interval_idx   = 0

    while updates_so_far < total_updates and not early_stop:
        step = min(pbt_interval - (updates_so_far % pbt_interval),
                   total_updates - updates_so_far)

        print(f"[agent {agent_idx}] {output_name}: "
              f"updates {updates_so_far}→{updates_so_far+step}  "
              f"layout={layout}  pool_size={len(clone_pool)}")

        # build per-worker jobs — each worker gets its own randomly shuffled
        # per-update schedule of (clone_name, role) pairs for this interval,
        # drawn with replacement from the full clone_pool
        jobs = []
        for worker in workers:
            # generate a shuffled schedule: `step` pairs drawn by repeating
            # and shuffling the pool so every pair appears roughly equally
            repeats  = (step + len(clone_pool) - 1) // len(clone_pool)
            schedule = (clone_pool * repeats)[:step]
            rng.shuffle(schedule)
            update_schedule = [tuple(p) for p in schedule]

            # pre-compute the matching score list for each update
            extra_vals_schedule = [
                clone_partner_scores(clone_scores, cn, role, extra_features)
                for cn, role in update_schedule
            ]

            csv_log = os.path.join(pbt_dir, f"worker_{worker.worker_idx:02d}.csv")

            jobs.append({
                "agent_idx":                   agent_idx,
                "worker_idx":                  worker.worker_idx,
                "hyperparams":                 worker.hyperparams,
                "checkpoint":                  worker.checkpoint,
                "opt_state":                   worker.opt_state,
                "rng_key":                     worker.rng_key,
                "clones_dir":                  clones_dir,
                "update_schedule":             update_schedule,
                "extra_feature_values_schedule": extra_vals_schedule,
                "layout":                      layout,
                "cook_time":                   cook_time,
                "step":                        step,
                "num_steps":                   num_steps,
                "ppo_epochs":                  ppo_epochs,
                "minibatch_size":              minibatch_size,
                "gamma":                       gamma,
                "vf_coef":                     vf_coef,
                "weight_decay":                weight_decay,
                "hidden_sizes":                hidden_sizes,
                "save_every":                  save_every,
                "seed":                        seed,
                "verbose":                     verbose,
                "shaping":                     shaping,
                "csv_log":                     csv_log,
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
            worker.update_count += step
            worker.opt_state = r["opt_state"]
            worker.rng_key   = r["rng_key"]

        updates_so_far += step

        at_pbt_boundary = (updates_so_far % pbt_interval == 0)
        if at_pbt_boundary:
            interval_idx += 1

        if at_pbt_boundary and updates_so_far < total_updates:
            workers = pbt_step(
                workers           = workers,
                rng               = rng,
                obs_dim           = obs_dim,
                hidden_sizes      = hidden_sizes,
                target_percentile = pbt_target_pct,
                perturb           = pbt_perturb,
                eval_window       = pbt_eval_window,
            )

        if at_pbt_boundary:
            for worker in workers:
                snap_path = os.path.join(
                    snapshots_dir,
                    f"worker_{worker.worker_idx:02d}_at_update_{updates_so_far:04d}.npz"
                )
                snap_params = load_params_npz(
                    worker.checkpoint,
                    like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
                )
                save_params_npz(snap_params, snap_path)
            best_snap = max(workers, key=lambda w: w.rolling_mean_reward(pbt_eval_window))
            print(f"[agent {agent_idx}] {output_name}: snapshots saved — "
                  f"all {n_workers} workers at update {updates_so_far} "
                  f"(best: worker {best_snap.worker_idx}, "
                  f"reward={best_snap.rolling_mean_reward(pbt_eval_window):.1f})")

            # save best-worker checkpoint at this interval boundary
            best_interval_ckpt = os.path.join(
                snapshots_dir,
                f"best_at_update_{updates_so_far:04d}.npz"
            )
            best_params = load_params_npz(
                best_snap.checkpoint,
                like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
            )
            save_params_npz(best_params, best_interval_ckpt)

            # evaluate best worker against every unique (clone, role) pair
            eval_csv = os.path.join(pbt_dir, "interval_eval.csv")
            evaluate_at_interval(
                best_worker    = best_snap,
                clone_pool     = clone_pool,
                clone_scores   = clone_scores,
                clones_dir     = clones_dir,
                extra_features = extra_features,
                layout         = layout,
                cook_time      = cook_time,
                hidden_sizes   = hidden_sizes,
                obs_dim        = obs_dim,
                updates_so_far = updates_so_far,
                eval_episodes  = eval_episodes,
                eval_steps     = eval_steps,
                eval_csv       = eval_csv,
                verbose        = verbose,
            )

            # ── validation clone evaluation + early stopping ──────────────
            if val_pool:
                val_csv = os.path.join(pbt_dir, "validation_eval.csv")
                val_results = evaluate_at_interval(
                    best_worker    = best_snap,
                    clone_pool     = val_pool,
                    clone_scores   = clone_scores,
                    clones_dir     = clones_dir,
                    extra_features = extra_features,
                    layout         = layout,
                    cook_time      = cook_time,
                    hidden_sizes   = hidden_sizes,
                    obs_dim        = obs_dim,
                    updates_so_far = updates_so_far,
                    eval_episodes  = eval_episodes,
                    eval_steps     = eval_steps,
                    eval_csv       = val_csv,
                    verbose        = verbose,
                )

                val_reward = float(np.mean([r["mean_reward"] for r in val_results]))
                improvement = val_reward - best_val_reward

                print(
                    f"[agent {agent_idx}] {output_name}: "
                    f"val_reward={val_reward:.2f}  "
                    f"best_so_far={best_val_reward:.2f}  "
                    f"improvement={improvement:.2f}  "
                    f"stuck_ctr={stuck_counter}/{stuck_patience}  "
                    f"plateau_ctr={plateau_counter}/{plateau_patience}"
                )

                # update best-val checkpoint if we improved at all
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    best_val_params = load_params_npz(
                        best_snap.checkpoint,
                        like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes),
                    )
                    save_params_npz(best_val_params, best_val_checkpoint)
                    print(f"[agent {agent_idx}] {output_name}: "
                          f"new best val checkpoint saved "
                          f"(reward={best_val_reward:.2f})")

                # update patience counters
                if improvement >= stuck_min_delta:
                    stuck_counter = 0
                else:
                    stuck_counter += 1

                if improvement >= plateau_min_delta:
                    plateau_counter = 0
                else:
                    plateau_counter += 1

                # check early stop conditions
                if stuck_counter >= stuck_patience:
                    print(
                        f"[agent {agent_idx}] {output_name}: EARLY STOP — "
                        f"no improvement >= {stuck_min_delta} for "
                        f"{stuck_patience} consecutive intervals "
                        f"(stuck condition)"
                    )
                    early_stop = True
                elif plateau_counter >= plateau_patience:
                    print(
                        f"[agent {agent_idx}] {output_name}: EARLY STOP — "
                        f"no improvement >= {plateau_min_delta} for "
                        f"{plateau_patience} consecutive intervals "
                        f"(plateau condition)"
                    )
                    early_stop = True

    # ── select best worker and save final checkpoint ─────────────────────────
    best_worker = max(workers, key=lambda w: w.rolling_mean_reward(pbt_eval_window))
    print(f"[agent {agent_idx}] {output_name}: best worker {best_worker.worker_idx}  "
          f"reward={best_worker.rolling_mean_reward(pbt_eval_window):.1f}  "
          f"hyperparams={best_worker.hyperparams}")

    if early_stop and os.path.exists(best_val_checkpoint):
        # use the best-validation checkpoint rather than the most recent one
        params = load_params_npz(
            best_val_checkpoint,
            like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
        )
        print(f"[agent {agent_idx}] {output_name}: "
              f"using best-val checkpoint (val_reward={best_val_reward:.2f})")
    else:
        params = load_params_npz(
            best_worker.checkpoint,
            like_params=init_model(obs_dim, ACT_DIM, hidden_sizes=hidden_sizes)
        )
    save_params_npz(params, final_checkpoint)

    result = {
        "status":           "complete",
        "agent_idx":        agent_idx,
        "output_name":      output_name,
        "final_checkpoint": final_checkpoint,
        "snapshots_dir":    snapshots_dir,
        "layout":           layout,
        "extra_features":   extra_features,
        "training_clones":  training_clones,
        "validation_clones": validation_clones,
        "obs_dim":          obs_dim,
        "best_worker":      best_worker.to_dict(),
        "all_workers":      [w.to_dict() for w in workers],
        "early_stop":       early_stop,
        "best_val_reward":  best_val_reward if val_pool else None,
        "updates_completed": updates_so_far,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[agent {agent_idx}] {output_name}: complete — {result_path}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="PBT training of one agent against frozen clone partners."
    )
    ap.add_argument("--agent-config", type=str, required=True,
                    help="Path to a JSON file containing one agent config "
                         "(one entry from agents.json, with defaults applied)")
    ap.add_argument("--clones-dir",    type=str, required=True,
                    help="Directory containing clone .npz checkpoints")
    ap.add_argument("--clones-scores", type=str, required=True,
                    help="Path to artifacts/partners/competence_labels.json")
    ap.add_argument("--out-dir",       type=str, required=True)
    ap.add_argument("--baselines-dir", type=str, default="baselines/",
                    help="Directory containing baseline .npz checkpoints "
                         "(default: baselines/)")
    ap.add_argument("--agent-idx",     type=int, default=0)

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
                    help="sequential: run PBT workers one at a time "
                         "(lowest memory, default). "
                         "parallel: run all workers concurrently via "
                         "multiprocessing (faster, more memory).")

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

    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.agent_config) as f:
        agent_cfg = json.load(f)

    clone_scores = load_clone_scores(args.clones_scores)

    train_agent(
        agent_cfg          = agent_cfg,
        clones_dir         = args.clones_dir,
        clone_scores       = clone_scores,
        out_dir            = args.out_dir,
        baselines_dir      = args.baselines_dir,
        agent_idx          = args.agent_idx,
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
        shaping            = {
            "pick_onion":   args.shaping_pick_onion,
            "place_onion":  args.shaping_place_onion,
            "pick_dish":    args.shaping_pick_dish,
            "deliver_soup": args.shaping_deliver_soup,
        },
        worker_parallelism = args.worker_parallelism,
        eval_episodes      = args.eval_episodes,
        eval_steps         = args.eval_steps,
    )


if __name__ == "__main__":
    main()