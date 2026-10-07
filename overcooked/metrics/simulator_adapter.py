from __future__ import annotations
INTERACT_ACTION = 5
KITCHEN_EVENTS = ['pick_onion', 'place_onion_counter', 'place_onion_pot', 'pick_plate', 'place_plate_counter', 'fill_plate', 'place_soup_counter', 'deliver_soup']
LAYOUT_NAME_MAP = {
    "Cramped_Room":        "cramped_room",
    "cramped_room":        "cramped_room",
    "Coord_Ring":          "coord_ring",
    "coord_ring":          "coord_ring",
    "coordination_ring":   "coord_ring",
    "Asymm_Advantages":    "asymm_advantages",
    "asymm_advantages":    "asymm_advantages",
    "asymmetric_advantages": "asymm_advantages",
    "Counter_Circuit":     "counter_circuit",
    "counter_circuit":     "counter_circuit",
    "Forced_Coordination": "forced_coord",
    "forced_coord":        "forced_coord",
    "forced_coordination": "forced_coord",
}

def detect_events_from_inv(
    act:             int,
    pre_inv:         str,    # "empty" | "onion" | "plate" | "soup"
    post_inv:        str,
    reward:          float,
    agent_pos:       Tuple[int, int],   # (x, y) pre-step position
    agent_dir:       Tuple[int, int],   # (dx, dy) facing direction
    pot_positions:   set,               # set of (x, y) pot cells
    serving_positions: set,             # set of (x, y) serving counter cells
) -> Dict[str, bool]:
    """
    Per-agent kitchen event detector using ground-truth inventory transitions
    and agent position/direction to disambiguate placement targets.

    Each event is determined by comparing pre_inv and post_inv (read directly
    from state.agent_inv before and after the step). For placement events
    (onion→empty, plate→empty, soup→empty) we use the agent's facing direction
    to determine which cell they interacted with, resolving pot vs counter
    without relying on global observation channel deltas.

    This is fully per-agent and immune to the global-channel-delta problem.
    """
    ev = {e: False for e in KITCHEN_EVENTS}

    if act != INTERACT_ACTION:
        return ev

    # cell the agent is facing / interacting with
    target_cell = (agent_pos[0] + agent_dir[0], agent_pos[1] + agent_dir[1])

    # ── delivery ──────────────────────────────────────────────────────────────
    # Must be facing a serving counter AND holding soup AND reward fired.
    # The spatial check makes this per-agent even if both receive the reward.
    if (reward > 0
            and pre_inv == "soup"
            and post_inv == "empty"
            and target_cell in serving_positions):
        ev["deliver_soup"] = True
        return ev

    # ── pickups (empty → something) ───────────────────────────────────────────
    if pre_inv == "empty" and post_inv == "onion":
        ev["pick_onion"] = True
        return ev

    if pre_inv == "empty" and post_inv == "plate":
        ev["pick_plate"] = True
        return ev

    # ── onion placements (onion → empty) ─────────────────────────────────────
    if pre_inv == "onion" and post_inv == "empty":
        if target_cell in pot_positions:
            ev["place_onion_pot"] = True
        else:
            ev["place_onion_counter"] = True
        return ev

    # ── fill plate (plate → soup) ─────────────────────────────────────────────
    if pre_inv == "plate" and post_inv == "soup":
        ev["fill_plate"] = True
        return ev

    # ── plate placed on counter (plate → empty) ───────────────────────────────
    if pre_inv == "plate" and post_inv == "empty":
        ev["place_plate_counter"] = True
        return ev

    # ── soup placed on counter (soup → empty, no reward) ─────────────────────
    if pre_inv == "soup" and post_inv == "empty" and reward == 0:
        ev["place_soup_counter"] = True
        return ev

    return ev
def _get_layout_stations(env) -> Dict[str, set]:
    """
    Extract all station positions from a JaxMARL environment object.
    Returns a dict of sets of (x, y) tuples for each station type.
    Call this once per layout at startup, not once per timestep.
    """
    W = env.width
    
    def idx_to_xy(indices) -> set:
        # convert flat grid indices to (x, y) tuples
        # flat index = y * W + x, so x = idx % W, y = idx // W
        return {(int(i) % W, int(i) // W) for i in indices}
    
    return {
        "pots":             idx_to_xy(env.layout["pot_idx"]),
        "onion_dispensers": idx_to_xy(env.layout["onion_pile_idx"]),
        "plate_dispensers": idx_to_xy(env.layout["plate_pile_idx"]),
        "serving_counters": idx_to_xy(env.layout["goal_idx"]),
    }
