"""
ppo_core.py — Shared PPO machinery used by bc_ppo_finetune.py
and adaptive_train.py.

Contains:
  - AdamW optimizer
  - GAE computation (JAX scan)
  - PPO loss
  - Action sampling utilities
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

# ── AdamW optimizer ───────────────────────────────────────────────────────────

@dataclass
class AdamState:
    m: Any
    v: Any
    t: int


def adam_init(params) -> AdamState:
    return AdamState(
        m=tree_map(jnp.zeros_like, params),
        v=tree_map(jnp.zeros_like, params),
        t=0,
    )


def adam_step(
    params,
    grads,
    opt:          AdamState,
    lr:           float = 1e-3,
    beta1:        float = 0.9,
    beta2:        float = 0.999,
    eps:          float = 1e-8,
    weight_decay: float = 1e-4,
) -> Tuple:
    """
    AdamW update step.
    Weight decay regularizes against overfitting — important for BC
    clones trained on small datasets, and for PPO stability.
    """
    t  = opt.t + 1
    m  = tree_map(lambda m, g: beta1*m + (1-beta1)*g,   opt.m, grads)
    v  = tree_map(lambda v, g: beta2*v + (1-beta2)*g*g, opt.v, grads)
    mh = tree_map(lambda x: x / (1 - beta1**t), m)
    vh = tree_map(lambda x: x / (1 - beta2**t), v)
    new_params = tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps) - weight_decay * p,
        params, mh, vh
    )
    return new_params, AdamState(m=m, v=v, t=t)

# ── Generalized Advantage Estimation ─────────────────────────────────────────

def compute_gae(
    rewards: jnp.ndarray,
    values:  jnp.ndarray,
    dones:   jnp.ndarray,
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GAE advantages and returns for a rollout.

    Parameters
    ----------
    rewards : (T,) float32 — reward at each timestep
    values  : (T,) float32 — value estimate at each timestep
    dones   : (T,) float32 — 1.0 if episode ended, 0.0 otherwise
    gamma   : discount factor
    lam     : GAE lambda — trades off bias vs variance
                lam=1.0 → full Monte Carlo (high variance, low bias)
                lam=0.0 → TD(0) (low variance, high bias)

    Returns
    -------
    advantages : (T,) float32
    returns    : (T,) float32 — used as value targets
    """
    T = rewards.shape[0]

    # next_values[t] = values[t+1] for t < T-1, 0.0 for t = T-1
    # mask[t] = 0 if episode ended at t (no bootstrap), 1 otherwise
    next_values = jnp.concatenate([values[1:], jnp.zeros((1,))])
    masks       = 1.0 - dones

    def _scan_step(gae, t):
        """Single backward step of GAE recurrence."""
        delta = rewards[t] + gamma * next_values[t] * masks[t] - values[t]
        gae   = delta + gamma * lam * masks[t] * gae
        return gae, gae

    # scan backward over timesteps
    _, advantages = jax.lax.scan(
        _scan_step,
        init   = jnp.zeros(()),       # initial gae = 0.0
        xs     = jnp.arange(T - 1, -1, -1),  # T-1 down to 0
        reverse = False,              # we manually reversed xs
    )

    # scan returns results in the order xs was traversed — reverse back
    advantages = advantages[::-1]
    returns    = advantages + values

    return advantages, returns

# ── action utilities ──────────────────────────────────────────────────────────

def cat_logprob(logits: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Log probability of chosen actions under a categorical distribution."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        log_probs, actions[..., None], axis=-1
    ).squeeze(-1)


def cat_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a categorical distribution."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs     = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


# ── PPO loss ──────────────────────────────────────────────────────────────────

def ppo_loss(
    params,
    batch:    dict,
    clip_eps: float = 0.2,
    vf_coef:  float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple:
    """
    PPO clipped surrogate loss.

    batch keys
    ----------
    obs      : (B, obs_dim)  — feature vectors
    act      : (B,)          — actions taken
    old_logp : (B,)          — log probs at collection time
    adv      : (B,)          — GAE advantages
    ret      : (B,)          — returns (value targets)

    Returns
    -------
    total_loss : scalar
    (pg_loss, vf_loss, entropy) : tuple of diagnostic scalars
    """
    from network import forward

    logits, values = forward(params, batch["obs"])

    # policy gradient loss — clipped surrogate objective
    logp  = cat_logprob(logits, batch["act"])
    ratio = jnp.exp(logp - batch["old_logp"])

    # normalise advantages within batch for stable training
    adv   = batch["adv"]
    # adv   = (adv - adv.mean()) / (adv.std() + 1e-8)

    pg_loss = -jnp.mean(jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    ))

    # value function loss
    vf_loss = 0.5 * jnp.mean((batch["ret"] - values) ** 2)

    # entropy bonus — encourages exploration / prevents collapse
    entropy = jnp.mean(cat_entropy(logits))

    total_loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

    return total_loss, (pg_loss, vf_loss, entropy)

# ── rollout collection ────────────────────────────────────────────────────────

def collect_rollout(
    params,
    env,
    obs,
    state,
    key:         jax.Array,
    stations:    dict,
    wall_map,
    grid_w:      int,
    grid_h:      int,
    cook_time:   int,
    n_steps:     int,
    player_idx:  int,
    partner_fn,
    prev_action: int = 4,
    shaping_cfg: dict = None,
    extra_feat_fn = None,
) -> Tuple:
    """
    Collect a rollout of n_steps transitions from the environment.

    Parameters
    ----------
    params      : network parameters for the learning agent
    env         : JaxMARL Overcooked environment
    obs         : current observations dict
    state       : current environment state
    key         : JAX random key
    stations    : output of _get_layout_stations()
    wall_map    : 2D boolean numpy array
    grid_w/h    : grid dimensions
    cook_time   : steps for pot to finish cooking
    n_steps     : number of steps to collect
    player_idx  : which agent is learning (0 or 1)
    partner_fn  : callable(obs, state, key) -> action for the partner
    prev_action : previous action for feature vector
    shaping_cfg : reward shaping weights dict (optional)
    extra_feat_fn : optional callable() -> list[float]. If provided, its
                    return value (an ordered list of scalars) is appended
                    to the learning agent's feature vector before it is
                    passed to the policy network. Used to condition the
                    learner on e.g. a frozen partner clone's skill scores.
                    The same values are appended at every step so the
                    stored obs has a consistent shape for PPO updates.

    Returns
    -------
    batch       : dict of stacked arrays for PPO update
    obs         : final observation
    state       : final state
    key         : updated random key
    prev_action : final prev_action for next rollout
    """
    from network import forward
    from bc_features import featurize_jaxmarl
    from reward_shaping import compute_shaping
    import numpy as np

    if shaping_cfg is None:
        shaping_cfg = {}

    agents     = env.agents
    self_agent = agents[player_idx]
    part_agent = agents[1 - player_idx]
    part_idx   = 1 - player_idx

    obs_buf   = []
    act_buf   = []
    logp_buf  = []
    val_buf   = []
    rew_buf   = []
    done_buf  = []

    prev_raw_obs = None   # for reward shaping

    # ── counter ownership tracking ────────────────────────────────────────────
    # Maps (row, col) -> who placed the item: 'partner' or 'self'
    # Used to reward counter pickups only when partner placed the item
    # OBJECT_TO_INDEX: empty=1, onion=3, plate=5
    INV_EMPTY = 1
    INV_ONION = 3
    INV_PLATE = 5

    partner_onion_cells: set = set()   # counter cells with partner-placed onions
    partner_dish_cells:  set = set()   # counter cells with partner-placed dishes

    def _find_ch11_change(prev_o, curr_o, decreased: bool):
        """Find counter cell (row,col) where ch11 changed direction."""
        if prev_o is None or curr_o is None:
            return None
        diff = curr_o[:, :, 11].astype(int) - prev_o[:, :, 11].astype(int)
        where = np.where(diff < 0) if decreased else np.where(diff > 0)
        if len(where[0]) == 0:
            return None
        return (int(where[0][0]), int(where[1][0]))

    for _ in range(n_steps):
        key, k_self, k_part, k_step = jax.random.split(key, 4)

        # capture inventories BEFORE step for ownership detection
        prev_self_inv = int(state.agent_inv[player_idx])
        prev_part_inv = int(state.agent_inv[part_idx])
        prev_obs_arr  = np.array(obs[self_agent])

        # featurize current observation for learning agent
        feat = featurize_jaxmarl(
            obs         = prev_obs_arr,
            state       = state,
            player_idx  = player_idx,
            stations    = stations,
            wall_map    = wall_map,
            grid_w      = grid_w,
            grid_h      = grid_h,
            cook_time   = cook_time,
            prev_action = prev_action,
        )
        feat_j = jnp.asarray(feat)

        # optionally append extra features (e.g. partner clone scores).
        # extra_feat_fn() returns a list of floats, one per extra feature,
        # in the order specified by agents.json extra_features.
        if extra_feat_fn is not None:
            extra_vals = extra_feat_fn()
            feat       = np.concatenate([feat, np.array(extra_vals, dtype=np.float32)])
            feat_j     = jnp.asarray(feat)

        # sample action from policy
        logits, value = forward(params, feat_j)
        action   = jax.random.categorical(k_self, logits)
        log_prob = cat_logprob(logits[None], action[None]).squeeze()

        # get partner action
        part_action = partner_fn(obs[part_agent], state, k_part)

        # build action dict
        actions = {
            self_agent: jnp.array(action, dtype=jnp.uint32),
            part_agent: jnp.array(part_action, dtype=jnp.uint32),
        }

        # step environment
        obs, state, rewards, dones, _ = env.step(k_step, state, actions)

        curr_obs_arr  = np.array(obs[self_agent])
        curr_self_inv = int(state.agent_inv[player_idx])
        curr_part_inv = int(state.agent_inv[part_idx])

        # ── update counter ownership ──────────────────────────────────────────

        # partner placed onion on counter (inv: onion→empty)
        if prev_part_inv == INV_ONION and curr_part_inv == INV_EMPTY:
            cell = _find_ch11_change(prev_obs_arr, curr_obs_arr, decreased=True)
            if cell is not None:
                partner_onion_cells.add(cell)

        # partner placed dish on counter (inv: plate→empty)
        if prev_part_inv == INV_PLATE and curr_part_inv == INV_EMPTY:
            cell = _find_ch11_change(prev_obs_arr, curr_obs_arr, decreased=True)
            if cell is not None:
                partner_dish_cells.add(cell)

        # ── detect counter pickup by self ─────────────────────────────────────
        picked_from_partner_onion = False
        picked_from_partner_dish  = False

        # self picked up onion — was it from counter or dispenser?
        if prev_self_inv == INV_EMPTY and curr_self_inv == INV_ONION:
            ch23_delta = float(curr_obs_arr[:, :, 23].sum()) - \
                         float(prev_obs_arr[:, :, 23].sum())
            if ch23_delta <= 0:   # counter pickup (dispenser would increase ch23)
                cell = _find_ch11_change(prev_obs_arr, curr_obs_arr, decreased=False)
                if cell is not None and cell in partner_onion_cells:
                    picked_from_partner_onion = True
                    partner_onion_cells.discard(cell)

        # self picked up dish — was it from counter or dispenser?
        if prev_self_inv == INV_EMPTY and curr_self_inv == INV_PLATE:
            ch22_delta = float(curr_obs_arr[:, :, 22].sum()) - \
                         float(prev_obs_arr[:, :, 22].sum())
            if ch22_delta <= 0:   # counter pickup
                cell = _find_ch11_change(prev_obs_arr, curr_obs_arr, decreased=False)
                if cell is not None and cell in partner_dish_cells:
                    picked_from_partner_dish = True
                    partner_dish_cells.discard(cell)

        # ── compute shaping ───────────────────────────────────────────────────
        raw_reward    = float(rewards[self_agent])
        shaping_bonus = compute_shaping(
            obs_t                    = curr_obs_arr,
            prev_obs_t               = prev_raw_obs,
            shaping_cfg              = shaping_cfg,
            picked_from_partner_onion = picked_from_partner_onion,
            picked_from_partner_dish  = picked_from_partner_dish,
        )
        prev_raw_obs = curr_obs_arr

        # record transition
        obs_buf.append(feat)
        act_buf.append(int(action))
        logp_buf.append(float(log_prob))
        val_buf.append(float(value))
        rew_buf.append(raw_reward + shaping_bonus)
        done_buf.append(float(dones.get("__all__", False)))

        prev_action = int(action)

        # reset if episode ended
        if dones.get("__all__", False):
            key, k_reset = jax.random.split(key)
            obs, state   = env.reset(k_reset)
            prev_action  = 4
            prev_raw_obs = None
            partner_onion_cells.clear()
            partner_dish_cells.clear()

    import numpy as np
    batch = {
        "obs":      jnp.asarray(np.stack(obs_buf)),
        "act":      jnp.asarray(np.array(act_buf,  dtype=np.int32)),
        "old_logp": jnp.asarray(np.array(logp_buf, dtype=np.float32)),
        "val":      jnp.asarray(np.array(val_buf,  dtype=np.float32)),
        "rew":      jnp.asarray(np.array(rew_buf,  dtype=np.float32)),
        "done":     jnp.asarray(np.array(done_buf, dtype=np.float32)),
    }

    return batch, obs, state, key, prev_action