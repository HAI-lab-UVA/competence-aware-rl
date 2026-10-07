"""
agent_finetune.py — Train an agent via PPO against frozen, pre-trained
behavioral clone partners, one per PPO update.

  - Learner policy is randomly initialised (no BC pre-training).
  - Partner is a frozen clone; weights never change (no periodic sync).
  - The partner clone AND its seat (clone_0/clone_1) can change every
    update according to a pre-generated update_schedule passed by the
    caller.
  - When clone plays clone_0 (seat 0), learner plays seat 1, and vice
    versa.
  - If cfg.extra_feature_values_schedule is set, each update uses the
    corresponding pre-computed score list for the active clone/role,
    appended to the learner's observation vector.
  - obs_dim = FEATURE_DIM + len(extra_features) (constant across all
    updates; the VALUES change per update as the partner rotates).

Metrics are accumulated in memory every update and flushed to CSV only
at the end of the interval (when the checkpoint is also saved), keeping
IO minimal.
"""
from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

from network import init_model, forward
from checkpoint import save_params_npz, load_params_npz
from bc_features import FEATURE_DIM, _get_layout_stations
from ppo_core import (
    adam_init, adam_step, compute_gae, ppo_loss,
    collect_rollout, cat_logprob, cat_entropy,
)
from agent_clones import (
    ClonePartner, clone_checkpoint_path,
    learner_player_idx, clone_player_idx,
)

ACT_DIM = 6

# CSV columns — written once as header, then one row per PPO update
CSV_COLUMNS = [
    "update",
    "total_steps",
    "clone_name",
    "clone_role",
    "learner_seat",
    # reward
    "mean_episode_reward",     # mean total reward per completed episode
    # loss components
    "total_loss",
    "policy_loss",             # PPO clipped surrogate loss
    "value_loss",              # critic MSE loss
    "entropy",                 # mean policy entropy (higher = more exploration)
    # PPO diagnostics (computed once per update over the full batch)
    "approx_kl",               # approximate KL between old and new policy
    "clip_fraction",           # fraction of ratios that were clipped
    "explained_variance",      # how well the critic predicts returns (1=perfect)
    # gradient diagnostics
    "mean_grad_norm",          # mean L2 norm of gradients across minibatches
    # timing
    "elapsed",
]


# ── diagnostic helpers ────────────────────────────────────────────────────────

def _tree_l2_norm(grads) -> float:
    """L2 norm of a pytree of gradients."""
    leaves = jax.tree_util.tree_leaves(grads)
    sq_sum = sum(float(jnp.sum(g ** 2)) for g in leaves)
    return float(np.sqrt(sq_sum))


def compute_diagnostics(params, ppo_batch: dict) -> dict:
    """
    Compute diagnostic metrics over the full post-update batch.
    Called once per PPO update after all minibatch gradient steps,
    using the updated params.

    Returns
    -------
    dict with approx_kl, clip_fraction, explained_variance
    """
    obs      = jnp.asarray(ppo_batch["obs"])
    act      = jnp.asarray(ppo_batch["act"])
    old_logp = jnp.asarray(ppo_batch["old_logp"])
    ret      = jnp.asarray(ppo_batch["ret"])

    logits, values = forward(params, obs)
    new_logp = cat_logprob(logits, act)
    ratio    = jnp.exp(new_logp - old_logp)

    # approximate KL: E[log(old/new)] ≈ E[(ratio - 1) - log(ratio)]
    approx_kl = float(jnp.mean((ratio - 1) - jnp.log(ratio + 1e-8)))

    # fraction of ratios outside [1-clip_eps, 1+clip_eps] — any clipping
    # regardless of direction; we don't need clip_eps here, just check
    # if ratio != clipped_ratio (approximated as |ratio - 1| > 0.2 threshold)
    # Better: just check how many were outside the clip range used in training
    # We don't have clip_eps here, so use |ratio - 1| > 0.2 as a proxy.
    clip_fraction = float(jnp.mean(jnp.abs(ratio - 1.0) > 0.2))

    # explained variance: 1 - Var(returns - values) / Var(returns)
    values_flat = values.squeeze()
    var_ret     = jnp.var(ret)
    var_res     = jnp.var(ret - values_flat)
    explained_variance = float(
        1.0 - var_res / (var_ret + 1e-8) if var_ret > 1e-8 else 0.0
    )

    return {
        "approx_kl":          approx_kl,
        "clip_fraction":      clip_fraction,
        "explained_variance": explained_variance,
    }


def _mean_episode_reward(rew: np.ndarray, done: np.ndarray) -> float:
    """
    Compute mean total reward per episode from a rollout buffer.
    rew  : (num_steps,) reward at each step
    done : (num_steps,) 1.0 where episode ended
    """
    episode_returns = []
    ep_return = 0.0
    for r, d in zip(rew, done):
        ep_return += float(r)
        if float(d) > 0.5:
            episode_returns.append(ep_return)
            ep_return = 0.0
    # include the partial episode at the end if it has any reward
    if ep_return != 0.0 or not episode_returns:
        episode_returns.append(ep_return)
    return float(np.mean(episode_returns))


# ── hyperparameter config ─────────────────────────────────────────────────────

@dataclass
class AgentFinetuneConfig:
    # environment
    layout:    str
    cook_time: int

    # training
    num_updates:    int
    num_steps:      int
    ppo_epochs:     int
    minibatch_size: int
    gamma:          float
    gae_lambda:     float
    clip_eps:       float
    vf_coef:        float
    ent_coef:       float
    lr:             float
    weight_decay:   float
    hidden_sizes:   Tuple[int, ...]

    # identity
    agent_idx:  int
    worker_idx: int

    # I/O
    out:        str
    csv_log:    Optional[str]
    save_every: int
    seed:       int
    verbose:    bool
    shaping:    dict = None

    # Per-update schedule of (clone_name, role) pairs, one per update.
    update_schedule: List[Tuple[str, str]] = field(default_factory=list)

    # Per-update extra feature values, parallel to update_schedule.
    extra_feature_values_schedule: List[List[float]] = field(default_factory=list)

    # Directory containing clone .npz checkpoints
    clones_dir: str = ""

    @property
    def n_extra_features(self) -> int:
        if self.extra_feature_values_schedule:
            return len(self.extra_feature_values_schedule[0])
        return 0

    @property
    def obs_dim(self) -> int:
        return FEATURE_DIM + self.n_extra_features


# ── fine-tuning loop ──────────────────────────────────────────────────────────

def finetune_vs_clone(
    cfg:           AgentFinetuneConfig,
    checkpoint_in: Optional[str] = None,
    opt_state              = None,
    rng_key_in             = None,
):
    """
    Train (or continue training) the learner via PPO. The partner clone
    and its seat are determined per-update from cfg.update_schedule.

    Metrics are accumulated in memory each update and written to
    cfg.csv_log only at save_every boundaries, minimising IO.

    Returns
    -------
    params, reward_history, opt_state, rng_key
    """
    assert len(cfg.update_schedule) == cfg.num_updates, (
        f"update_schedule length {len(cfg.update_schedule)} != "
        f"num_updates {cfg.num_updates}"
    )
    if cfg.extra_feature_values_schedule:
        assert len(cfg.extra_feature_values_schedule) == cfg.num_updates

    obs_dim = cfg.obs_dim

    # ── RNG key ──────────────────────────────────────────────────────────────
    key = rng_key_in if rng_key_in is not None else (
        jax.random.PRNGKey(cfg.seed + cfg.worker_idx)
    )

    # ── environment ───────────────────────────────────────────────────────────
    env        = make("overcooked", layout=overcooked_layouts[cfg.layout])
    key, k_r   = jax.random.split(key)
    obs, state = env.reset(k_r)
    wall_map   = np.array(state.wall_map)
    stations   = _get_layout_stations(env)

    # ── learner params ────────────────────────────────────────────────────────
    if checkpoint_in is not None and os.path.exists(checkpoint_in):
        like_params = init_model(obs_dim, ACT_DIM,
                                  hidden_sizes=cfg.hidden_sizes,
                                  key=jax.random.PRNGKey(cfg.seed))
        params = load_params_npz(checkpoint_in, like_params=like_params)
        if cfg.verbose:
            print(f"[agent_finetune] loaded checkpoint: {checkpoint_in}")
    else:
        init_key = jax.random.fold_in(
            jax.random.PRNGKey(cfg.seed), cfg.worker_idx
        )
        params = init_model(obs_dim, ACT_DIM,
                             hidden_sizes=cfg.hidden_sizes,
                             key=init_key)
        if cfg.verbose:
            print(f"[agent_finetune] random init (obs_dim={obs_dim})")

    opt = opt_state if opt_state is not None else adam_init(params)

    # ── clone partner cache ───────────────────────────────────────────────────
    unique_clones = {name for name, _ in cfg.update_schedule}
    partners: Dict[str, ClonePartner] = {}
    for clone_name in unique_clones:
        ckpt = clone_checkpoint_path(cfg.clones_dir, clone_name)
        partners[clone_name] = ClonePartner(
            name         = clone_name,
            checkpoint   = ckpt,
            hidden_sizes = cfg.hidden_sizes,
            seed         = cfg.seed,
        )

    # ── in-memory metric buffer ───────────────────────────────────────────────
    # Each entry is one row dict (one per PPO update). Flushed to CSV at
    # save_every boundaries to minimise IO.
    metric_buffer: List[dict] = []

    # ── training loop ─────────────────────────────────────────────────────────
    total_steps    = 0
    prev_action    = 4
    start          = time.time()
    reward_history: list = []

    loss_fn = jax.jit(jax.value_and_grad(ppo_loss, has_aux=True))

    for update_i, (clone_name, clone_role) in enumerate(cfg.update_schedule):
        update = update_i + 1

        l_player_idx = learner_player_idx(clone_role)
        c_player_idx = clone_player_idx(clone_role)

        partner_fn = partners[clone_name].make_partner_fn(
            stations   = stations,
            wall_map   = wall_map,
            grid_w     = env.width,
            grid_h     = env.height,
            player_idx = c_player_idx,
            cook_time  = cfg.cook_time,
        )

        extra_vals = (
            cfg.extra_feature_values_schedule[update_i]
            if cfg.extra_feature_values_schedule else []
        )
        extra_feat_fn = None
        if extra_vals:
            _vals = list(extra_vals)
            def extra_feat_fn(_v=_vals):
                return _v

        batch, obs, state, key, prev_action = collect_rollout(
            params        = params,
            env           = env,
            obs           = obs,
            state         = state,
            key           = key,
            stations      = stations,
            wall_map      = wall_map,
            grid_w        = env.width,
            grid_h        = env.height,
            cook_time     = cfg.cook_time,
            n_steps       = cfg.num_steps,
            player_idx    = l_player_idx,
            partner_fn    = partner_fn,
            prev_action   = prev_action,
            shaping_cfg   = cfg.shaping or {},
            extra_feat_fn = extra_feat_fn,
        )
        total_steps += cfg.num_steps

        advs, rets = compute_gae(
            rewards = batch["rew"],
            values  = batch["val"],
            dones   = batch["done"],
            gamma   = cfg.gamma,
            lam     = cfg.gae_lambda,
        )

        ppo_batch = {
            "obs":      batch["obs"],
            "act":      batch["act"],
            "old_logp": batch["old_logp"],
            "adv":      advs,
            "ret":      rets,
        }

        N   = cfg.num_steps
        idx = np.arange(N)
        losses, pgs, vfs, ents, grad_norms = [], [], [], [], []

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, N, cfg.minibatch_size):
                e  = min(s + cfg.minibatch_size, N)
                mb = {k: v[idx[s:e]] for k, v in ppo_batch.items()}
                (loss_val, (pg, vf, ent)), grads = loss_fn(
                    params, mb, cfg.clip_eps, cfg.vf_coef, cfg.ent_coef
                )
                grad_norms.append(_tree_l2_norm(grads))
                params, opt = adam_step(
                    params, grads, opt,
                    lr=cfg.lr, weight_decay=cfg.weight_decay
                )
                losses.append(float(loss_val))
                pgs.append(float(pg))
                vfs.append(float(vf))
                ents.append(float(ent))

        # diagnostics computed once per update over full batch with updated params
        diag = compute_diagnostics(params, ppo_batch)

        mean_ep_reward = _mean_episode_reward(batch["rew"], batch["done"])
        reward_history.append(mean_ep_reward)
        elapsed = time.time() - start

        if cfg.verbose:
            print(f"[agent_finetune] update {update:4d}/{cfg.num_updates}  "
                  f"loss={np.mean(losses):.4f}  "
                  f"reward={mean_ep_reward:.2f}  "
                  f"kl={diag['approx_kl']:.4f}  "
                  f"ev={diag['explained_variance']:.3f}  "
                  f"partner={clone_name}({clone_role})  "
                  f"seat={l_player_idx}  "
                  f"t={elapsed:.1f}s")

        # accumulate row in memory
        metric_buffer.append({
            "update":              update,
            "total_steps":         total_steps,
            "clone_name":          clone_name,
            "clone_role":          clone_role,
            "learner_seat":        l_player_idx,
            "mean_episode_reward": round(mean_ep_reward, 4),
            "total_loss":          round(float(np.mean(losses)), 6),
            "policy_loss":         round(float(np.mean(pgs)),    6),
            "value_loss":          round(float(np.mean(vfs)),    6),
            "entropy":             round(float(np.mean(ents)),   6),
            "approx_kl":           round(diag["approx_kl"],          6),
            "clip_fraction":       round(diag["clip_fraction"],       4),
            "explained_variance":  round(diag["explained_variance"],  4),
            "mean_grad_norm":      round(float(np.mean(grad_norms)),  6),
            "elapsed":             round(elapsed, 2),
        })

        # flush metrics + save checkpoint at save_every boundaries
        at_save = (update % cfg.save_every == 0) or (update == cfg.num_updates)
        if at_save and metric_buffer and cfg.csv_log:
            os.makedirs(os.path.dirname(cfg.csv_log) or ".", exist_ok=True)
            write_header = not os.path.exists(cfg.csv_log)
            with open(cfg.csv_log, "a", newline="") as cf:
                cw = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
                if write_header:
                    cw.writeheader()
                cw.writerows(metric_buffer)
            metric_buffer.clear()

        if at_save:
            os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
            save_params_npz(params, cfg.out)
            if cfg.verbose:
                print(f"[agent_finetune] saved -> {cfg.out}")

    if cfg.verbose:
        print(f"[agent_finetune] done -> {cfg.out}")

    return params, reward_history, opt, key
