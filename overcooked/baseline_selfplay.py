"""
baseline_selfplay.py — Train a baseline model from random init via self-play
PPO for a single layout.

Unlike bc_ppo_finetune.py, there is no BC pre-training checkpoint — the model
starts from a random initialisation. The model trains as BOTH seats, alternating
player_idx each PPO update (seat 0 on odd updates, seat 1 on even updates). The
frozen partner is a copy of the current weights, synced every
partner_update_every updates.

Only base features (FEATURE_DIM) are used — no extra_features. The saved
checkpoint is therefore compatible with FEATURE_DIM as obs_dim and can be
zero-extended for finetuning agents that use extra features.

Designed to be called by baseline_population.py which manages PBT across
multiple workers for a single layout.

Returns
-------
params, reward_history, opt_state, partner_params, rng_key
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl import make
from jaxmarl.environments.overcooked.layouts import overcooked_layouts

from network import init_model, forward
from checkpoint import save_params_npz, load_params_npz
from bc_features import FEATURE_DIM, _get_layout_stations, featurize_jaxmarl
from ppo_core import (
    adam_init, adam_step,
    compute_gae, ppo_loss,
    collect_rollout,
    cat_logprob, cat_entropy,
)

ACT_DIM = 6

CSV_COLUMNS = [
    "update",
    "total_steps",
    "seat",
    "mean_episode_reward",
    "total_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "mean_grad_norm",
    "elapsed",
]


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class BaselineSelfplayConfig:
    # environment
    layout:    str
    cook_time: int

    # training
    num_updates:          int
    num_steps:            int
    ppo_epochs:           int
    minibatch_size:       int
    gamma:                float
    gae_lambda:           float
    clip_eps:             float
    vf_coef:              float
    ent_coef:             float
    lr:                   float
    weight_decay:         float
    hidden_sizes:         Tuple[int, ...]
    partner_update_every: int

    # identity
    worker_idx: int

    # I/O
    out:        str
    csv_log:    Optional[str]
    save_every: int
    seed:       int
    verbose:    bool
    shaping:    dict = None

    @property
    def obs_dim(self) -> int:
        return FEATURE_DIM


# ── diagnostic helpers ────────────────────────────────────────────────────────

def _tree_l2_norm(grads) -> float:
    leaves = jax.tree_util.tree_leaves(grads)
    sq_sum = sum(float(jnp.sum(g ** 2)) for g in leaves)
    return float(np.sqrt(sq_sum))


def _compute_diagnostics(params, ppo_batch: dict) -> dict:
    obs      = jnp.asarray(ppo_batch["obs"])
    act      = jnp.asarray(ppo_batch["act"])
    old_logp = jnp.asarray(ppo_batch["old_logp"])
    ret      = jnp.asarray(ppo_batch["ret"])

    logits, values = forward(params, obs)
    new_logp = cat_logprob(logits, act)
    ratio    = jnp.exp(new_logp - old_logp)

    approx_kl     = float(jnp.mean((ratio - 1) - jnp.log(ratio + 1e-8)))
    clip_fraction = float(jnp.mean(jnp.abs(ratio - 1.0) > 0.2))

    values_flat = values.squeeze()
    var_ret     = jnp.var(ret)
    var_res     = jnp.var(ret - values_flat)
    explained_variance = float(
        1.0 - var_res / (var_ret + 1e-8) if var_ret > 1e-8 else 0.0
    )

    return {
        "approx_kl":         approx_kl,
        "clip_fraction":     clip_fraction,
        "explained_variance": explained_variance,
    }


def _mean_episode_reward(rew: np.ndarray, done: np.ndarray) -> float:
    episode_returns = []
    ep_return = 0.0
    for r, d in zip(rew, done):
        ep_return += float(r)
        if float(d) > 0.5:
            episode_returns.append(ep_return)
            ep_return = 0.0
    if ep_return != 0.0 or not episode_returns:
        episode_returns.append(ep_return)
    return float(np.mean(episode_returns))


# ── self-play training loop ───────────────────────────────────────────────────

def selfplay_baseline(
    cfg:               BaselineSelfplayConfig,
    checkpoint_in:     Optional[str] = None,
    opt_state                        = None,
    partner_params_in                = None,
    rng_key_in                       = None,
):
    """
    Train a baseline model from scratch (or resume) via self-play PPO.

    The model alternates seats each PPO update:
      - Even updates (0, 2, 4, ...): learner plays seat 0, partner plays seat 1
      - Odd  updates (1, 3, 5, ...): learner plays seat 1, partner plays seat 0

    The frozen partner is synced to the current weights every
    cfg.partner_update_every updates.

    Parameters
    ----------
    cfg               : BaselineSelfplayConfig
    checkpoint_in     : path to existing checkpoint to resume from, or None
    opt_state         : AdamState from a previous call, or None
    partner_params_in : partner policy params from a previous call, or None
    rng_key_in        : JAX PRNGKey from a previous call, or None

    Returns
    -------
    params, reward_history, opt_state, partner_params, rng_key
    """
    key = rng_key_in if rng_key_in is not None else (
        jax.random.PRNGKey(cfg.seed + cfg.worker_idx)
    )

    # ── environment ──────────────────────────────────────────────────────────
    env        = make("overcooked", layout=overcooked_layouts[cfg.layout])
    key, k_r   = jax.random.split(key)
    obs, state = env.reset(k_r)
    wall_map   = np.array(state.wall_map)
    stations   = _get_layout_stations(env)

    # ── learner params ────────────────────────────────────────────────────────
    like_params = init_model(
        FEATURE_DIM, ACT_DIM,
        hidden_sizes=cfg.hidden_sizes,
        key=jax.random.PRNGKey(cfg.seed),
    )

    if checkpoint_in is not None and os.path.exists(checkpoint_in):
        params = load_params_npz(checkpoint_in, like_params=like_params)
        if cfg.verbose:
            print(f"[baseline_selfplay] loaded checkpoint: {checkpoint_in}")
    else:
        init_key = jax.random.fold_in(
            jax.random.PRNGKey(cfg.seed), cfg.worker_idx
        )
        params = init_model(
            FEATURE_DIM, ACT_DIM,
            hidden_sizes=cfg.hidden_sizes,
            key=init_key,
        )
        if cfg.verbose:
            print(f"[baseline_selfplay] random init (obs_dim={FEATURE_DIM})")

    partner_params = (
        partner_params_in if partner_params_in is not None else params
    )
    opt = opt_state if opt_state is not None else adam_init(params)

    # ── partner function factory ──────────────────────────────────────────────
    def _make_partner_fn(partner_p, partner_seat: int):
        """
        Build a partner_fn(obs, state, key) -> action for the frozen partner
        occupying partner_seat.
        """
        _p    = partner_p
        _seat = partner_seat

        def partner_fn(partner_obs, partner_state, partner_key):
            feat = featurize_jaxmarl(
                obs        = np.array(partner_obs),
                state      = partner_state,
                player_idx = _seat,
                stations   = stations,
                wall_map   = wall_map,
                grid_w     = env.width,
                grid_h     = env.height,
                cook_time  = cfg.cook_time,
            )
            logits, _ = forward(_p, jnp.asarray(feat))
            return jax.random.categorical(partner_key, logits)

        return partner_fn

    # ── CSV logging ───────────────────────────────────────────────────────────
    metric_buffer: List[dict] = []

    # ── training loop ─────────────────────────────────────────────────────────
    total_steps    = 0
    prev_action    = 4
    start          = time.time()
    reward_history: List[float] = []

    loss_fn = jax.jit(jax.value_and_grad(ppo_loss, has_aux=True))

    for update_i in range(cfg.num_updates):
        update = update_i + 1

        # alternate seats: even update_i → seat 0, odd → seat 1
        learner_seat = update_i % 2
        partner_seat = 1 - learner_seat

        partner_fn = _make_partner_fn(partner_params, partner_seat)

        batch, obs, state, key, prev_action = collect_rollout(
            params      = params,
            env         = env,
            obs         = obs,
            state       = state,
            key         = key,
            stations    = stations,
            wall_map    = wall_map,
            grid_w      = env.width,
            grid_h      = env.height,
            cook_time   = cfg.cook_time,
            n_steps     = cfg.num_steps,
            player_idx  = learner_seat,
            partner_fn  = partner_fn,
            prev_action = prev_action,
            shaping_cfg = cfg.shaping or {},
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
                    lr=cfg.lr, weight_decay=cfg.weight_decay,
                )
                losses.append(float(loss_val))
                pgs.append(float(pg))
                vfs.append(float(vf))
                ents.append(float(ent))

        # sync partner every K updates
        if update % cfg.partner_update_every == 0:
            partner_params = params
            if cfg.verbose:
                print(f"[baseline_selfplay] synced partner at update {update}")

        diag           = _compute_diagnostics(params, ppo_batch)
        mean_ep_reward = _mean_episode_reward(
            np.array(batch["rew"]), np.array(batch["done"])
        )
        reward_history.append(mean_ep_reward)
        elapsed = time.time() - start

        if cfg.verbose:
            print(
                f"[baseline_selfplay] update {update:4d}/{cfg.num_updates}  "
                f"seat={learner_seat}  "
                f"loss={np.mean(losses):.4f}  "
                f"reward={mean_ep_reward:.2f}  "
                f"kl={diag['approx_kl']:.4f}  "
                f"ev={diag['explained_variance']:.3f}  "
                f"t={elapsed:.1f}s"
            )

        metric_buffer.append({
            "update":              update,
            "total_steps":         total_steps,
            "seat":                learner_seat,
            "mean_episode_reward": round(mean_ep_reward, 4),
            "total_loss":          round(float(np.mean(losses)), 6),
            "policy_loss":         round(float(np.mean(pgs)),    6),
            "value_loss":          round(float(np.mean(vfs)),    6),
            "entropy":             round(float(np.mean(ents)),   6),
            "approx_kl":           round(diag["approx_kl"],           6),
            "clip_fraction":       round(diag["clip_fraction"],        4),
            "explained_variance":  round(diag["explained_variance"],   4),
            "mean_grad_norm":      round(float(np.mean(grad_norms)),   6),
            "elapsed":             round(elapsed, 2),
        })

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
                print(f"[baseline_selfplay] saved -> {cfg.out}")

    if cfg.verbose:
        print(f"[baseline_selfplay] done -> {cfg.out}")

    return params, reward_history, opt, partner_params, key