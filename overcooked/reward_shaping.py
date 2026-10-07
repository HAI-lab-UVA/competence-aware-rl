"""
reward_shaping.py
Compute custom shaping rewards from Overcooked observation tensors.

Verified channel mappings (cramped_room, JaxMARL):
    CH_ONION       = 23  onion — increases ONLY on dispenser pickup
    CH_DISH        = 22  plate/dish — increases ONLY on dispenser pickup
    CH_SOUP        = 21  ready soup / plated soup
    CH_POT         = 10  pot location (static mask)
    CH_POT_ONIONS  = 16  onion count in pot (+1 for 1st/2nd onion placed)
    CH_POT_COOKING = 18  cooking state (increases when 3rd onion placed)
    CH_COUNTER     = 11  counter occupancy (decreases when object placed)

Spam protection:
    pick_onion/pick_dish from dispenser: ch23/ch22 only increase on
        dispenser pickup — counter cycling does NOT trigger these channels.
    pick_onion/pick_dish from counter: only rewarded if the PARTNER agent
        placed the object there (tracked via inventory in collect_rollout).
    place_onion: only fires at pot cells (ch16/ch18 at pot mask).
    plate_soup: only fires when ch21 decreases at pot cell.
"""

import numpy as np


def compute_shaping(
    obs_t,
    prev_obs_t,
    shaping_cfg,
    picked_from_partner_onion: bool = False,
    picked_from_partner_dish:  bool = False,
):
    """
    Compute shaping reward from consecutive observations.

    Parameters
    ----------
    obs_t                    : current obs (H,W,C) numpy/JAX array
    prev_obs_t               : previous obs (H,W,C)
    shaping_cfg              : dict of reward weights
    picked_from_partner_onion: True if agent picked onion from counter
                               placed by the partner this step
    picked_from_partner_dish : True if agent picked dish from counter
                               placed by the partner this step

    Returns
    -------
    float shaping reward
    """
    if prev_obs_t is None:
        return 0.0

    obs_t      = np.asarray(obs_t,      dtype=np.int32)
    prev_obs_t = np.asarray(prev_obs_t, dtype=np.int32)

    H, W, C = obs_t.shape
    reward  = 0.0

    def delta(ch):
        if ch >= C:
            return 0.0
        return float(obs_t[:, :, ch].sum() - prev_obs_t[:, :, ch].sum())

    def delta_at_pot(ch):
        """Change in channel ch restricted to pot cells (ch10 mask)."""
        if C <= 10 or ch >= C:
            return delta(ch)
        pot_mask = obs_t[:, :, 10] > 0
        if not pot_mask.any():
            return 0.0
        before = float(prev_obs_t[:, :, ch][pot_mask].sum())
        after  = float(obs_t[:, :, ch][pot_mask].sum())
        return after - before

    # ── pick onion ────────────────────────────────────────────────────────────
    # Dispenser pickup: ch23 increases (counter cycling doesn't trigger this)
    # Counter pickup:   only reward if partner placed it (passed as flag)
    if "pick_onion" in shaping_cfg:
        if delta(23) > 0:                    # dispenser pickup
            reward += shaping_cfg["pick_onion"]
        elif picked_from_partner_onion:      # partner's counter placement
            reward += shaping_cfg["pick_onion"]

    # ── place onion in pot ────────────────────────────────────────────────────
    # ch16 increases for 1st/2nd onion, ch18 increases for 3rd (cooking start)
    # Both checked at pot cells only — cannot be triggered elsewhere
    if "place_onion" in shaping_cfg:
        if delta_at_pot(16) > 0 or delta_at_pot(18) > 0:
            reward += shaping_cfg["place_onion"]

    # ── pick dish ─────────────────────────────────────────────────────────────
    # Same logic as onion
    if "pick_dish" in shaping_cfg:
        if delta(22) > 0:                   # dispenser pickup
            reward += shaping_cfg["pick_dish"]
        elif picked_from_partner_dish:      # partner's counter placement
            reward += shaping_cfg["pick_dish"]

    # ── plate soup (pick ready soup from pot) ─────────────────────────────────
    # ch21 decreases at pot cell when agent picks up ready soup
    if "plate_soup" in shaping_cfg and delta_at_pot(21) < 0:
        reward += shaping_cfg["plate_soup"]

    # ── deliver soup ──────────────────────────────────────────────────────────
    # ch21 decreases outside pot = delivery (distinct from plate_soup above)
    if "deliver_soup" in shaping_cfg:
        total_21  = delta(21)
        pot_21    = delta_at_pot(21)
        non_pot   = total_21 - pot_21
        if non_pot < 0:
            reward += shaping_cfg["deliver_soup"]

    # ── step cost ─────────────────────────────────────────────────────────────
    if "step_cost" in shaping_cfg:
        reward += shaping_cfg["step_cost"]

    return reward