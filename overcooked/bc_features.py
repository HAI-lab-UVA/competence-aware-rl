from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Mapping between Overcooked V1 & V2 names for the layouts
LAYOUT_NAME_MAP = {
    "asymmetric_advantages": "asymm_advantages",
    "forced_coordination": "forced_coord",
    "cramped_room": "cramped_room",
    "coordination_ring":"coord_ring",
    "counter_circuit": "counter_circuit"
}

# Feature Breakdown
_SELF_BLOCK = 26 # held item (4) + facing (2) + adjacency (16) + wall_adjacent (4)
_PARTNER_BLOCK = 35 # held item (4) + facing (2) + relative position (2) + reachable (1) + adjacency (16) + wall_adjacent (4) + previous action (6)
_PROXIMITY = 21 # 7 targets (dispeners, delivery, pot, items) x 3 features
_POT_STATES = 8 # 2 pots (onions, cook, ready, reachable)
_COUNTERS = 3 # Number of onions, dishes, and soups

# Number of Features
FEATURE_DIM = _SELF_BLOCK + _PARTNER_BLOCK + _PROXIMITY + _POT_STATES + _COUNTERS

# maps OBJECT_TO_INDEX inventory values to our held one-hot indices
_INV_TO_HELD_IDX = {
    1: 0,  # empty  → nothing
    3: 1,  # onion  → onion
    5: 2,  # plate  → dish
    9: 3,  # dish (plated soup) → soup
}

def _held_onehot_from_inv(inv_val: int) -> np.ndarray:
    """Convert a JaxMARL agent_inv integer to a 4-dim one-hot."""
    v = np.zeros(4, dtype=np.float32)
    v[_INV_TO_HELD_IDX.get(int(inv_val), 0)] = 1.0
    return v

# Flood fill function -- identify what objects the agent can actually reach
def _flood_fill(start: Tuple[int, int], wall_map: np.ndarray) -> set:
    visted = set()
    queue = deque([start])
    H, W = wall_map.shape 

    # BFS to identify where the agent can actually reach
    while queue:
        x, y = queue.popleft()

        # Skip if we have already visted this cell
        if(x, y) in visted:
            continue 

        # Skip if this cell is a counter -- the agent can't walk here
        if wall_map[y, x]:
            continue

        # This cell is reachable. Add its neighbors to the queue
        visted.add((x, y))
        queue.extend([
            (x+1, y), (x-1, y), (x, y+1), (x, y-1)
        ])

    
    return visted


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


# Returns the shortest path from start to any neighbor of target using only reachable cells. 
def _bfs_distance(
    start:     Tuple[int, int],
    target:    Tuple[int, int],
    reachable: set,
) -> float:
    neighbors = [
        (target[0]+1, target[1]), (target[0]-1, target[1]),
        (target[0], target[1]+1), (target[0], target[1]-1)
    ]
    reachable_neighbors = [n for n in neighbors if n in reachable]
    
    if not reachable_neighbors:
        return float("inf")
    
    visited = {(int(start[0]), int(start[1]))}
    queue = deque([((int(start[0]), int(start[1])), 0)])
    
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) in reachable_neighbors:
            return float(dist)
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            nx, ny = int(nx), int(ny)
            if (nx, ny) in reachable and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    
    return float("inf")



def _nearest(
    agent_pos:  Tuple[int, int],
    candidates: List[Tuple[int, int]],
    reachable:  set,
    grid_w:     int,
    grid_h:     int,
) -> Tuple[float, float, float]:
    """
    Find the nearest reachable station from a list of candidates using
    BFS path length. Returns (dx, dy, is_reachable) where:
        dx, dy       : signed offset normalized by grid size, in [-1, 1]
        is_reachable : 1.0 if reachable, 0.0 if not
    
    If candidates is empty or none are reachable, returns (-2, -2, 0)
    sentinel — outside the [-1, 1] range so the network can distinguish
    this from a real offset.
    """
    SENTINEL = (-2.0, -2.0, 0.0)
    
    if not candidates:
        return SENTINEL
    
    ax, ay = agent_pos
    
    # find nearest candidate by BFS path length
    best = None
    best_dist = float("inf")
    
    for candidate in candidates:
        dist = _bfs_distance(agent_pos, candidate, reachable)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    
    # no candidate was reachable
    if best is None or best_dist == float("inf"):
        return SENTINEL
    
    cx, cy = best
    
    # normalize offsets to [-1, 1] range
    dx = (cx - ax) / (grid_w - 1)
    dy = (cy - ay) / (grid_h - 1)
    
    return (dx, dy, 1.0)


def _adjacency_flags(
    agent_pos: Tuple[int, int],
    stations:  Dict[str, List[Tuple[int, int]]],
    wall_map:  np.ndarray,
) -> np.ndarray:
    """
    Returns a 20-dim binary array indicating what is directly adjacent
    to the agent in each of the 4 directions.

    Layout: [up, down, left, right] x [pot, onion, plate, serving]  (16 bits)
          + [up, down, left, right] wall/out-of-bounds                (4 bits)

    The wall bits let the agent distinguish plain impassable counters
    from open floor — without them, both directions look identical in
    the station block (all zeros).
    """
    flags = np.zeros(20, dtype=np.float32)

    ax, ay = agent_pos
    H, W   = wall_map.shape

    # directions in order: up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # station types in order — must match the ordering above
    station_types = ["pots", "onion_dispensers",
                     "plate_dispensers", "serving_counters"]

    for d_idx, (dx, dy) in enumerate(directions):
        nx, ny   = ax + dx, ay + dy
        neighbor = (nx, ny)

        # station bits (indices 0–15)
        for s_idx, stype in enumerate(station_types):
            if neighbor in stations[stype]:
                flags[d_idx * 4 + s_idx] = 1.0

        # wall bit (indices 16–19): out-of-bounds or wall cell
        if nx < 0 or ny < 0 or nx >= W or ny >= H or wall_map[ny, nx]:
            flags[16 + d_idx] = 1.0

    return flags

def _held_onehot(held_object: Optional[Dict]) -> np.ndarray:
    """
    Returns a 4-dim one-hot vector for the object an agent is holding.
    Indices: 0=nothing, 1=onion, 2=plate, 3=soup
    
    held_object is either None (holding nothing) or a dict with a "name" key
    from the JSON log format e.g. {"name": "onion"}.
    """
    v = np.zeros(4, dtype=np.float32)
    
    if held_object is None:
        v[0] = 1.0
    elif held_object["name"] == "onion":
        v[1] = 1.0
    elif held_object["name"] == "dish":
        v[2] = 1.0
    elif held_object["name"] == "soup":
        v[3] = 1.0
    else:
        # unknown object — default to nothing
        v[0] = 1.0
    
    return v

def featurize_symbolic(
    obs_dict:    Dict[str, Any],
    player_idx:  int,
    stations:    Dict[str, set],
    wall_map:    np.ndarray,
    grid_w:      int,
    grid_h:      int,
    cook_time:   int = 20,
    prev_action: int = 4,
) -> np.ndarray:
    """
    Convert one symbolic observation from a V1 JSON log to an 85-dim
    feature vector.

    Parameters
    ----------
    obs_dict    : one entry from ep_observations[episode][step]
    player_idx  : index of the player we are cloning (0 or 1)
    stations    : output of _get_layout_stations()
    wall_map    : 2D boolean numpy array, True = wall
    grid_w/h    : grid dimensions
    cook_time   : steps for a pot to finish cooking
    prev_action : the action taken at the previous timestep (default stay)
    """
    feat    = np.zeros(FEATURE_DIM, dtype=np.float32)
    players = obs_dict["players"]

    self_p  = players[player_idx]
    other_p = players[1 - player_idx]

    ax, ay = self_p["position"]
    
    # compute reachable cells for self and other from current positions
    self_reachable  = _flood_fill((int(ax), int(ay)), wall_map)
    ox, oy = other_p["position"]
    other_reachable = _flood_fill((int(ox), int(oy)), wall_map)

    idx = 0  # running index into feat

    # ── self held object (4) ─────────────────────────────────────────────
    feat[idx:idx+4] = _held_onehot(self_p.get("held_object"))
    idx += 4
    # print(f"after self held: idx={idx}")

    # ── self facing direction (2) ────────────────────────────────────────
    feat[idx:idx+2] = self_p["orientation"]
    idx += 2
    # print(f"after self facing: idx={idx}")

    # ── self adjacency flags (20) ────────────────────────────────────────
    feat[idx:idx+20] = _adjacency_flags((ax, ay), stations, wall_map)
    idx += 20
    # print(f"after self adjacency: idx={idx}")

    # ── partner held object (4) ──────────────────────────────────────────
    feat[idx:idx+4] = _held_onehot(other_p.get("held_object"))
    idx += 4
    # print(f"after partner held: idx={idx}")

    # ── partner facing direction (2) ─────────────────────────────────────
    feat[idx:idx+2] = other_p["orientation"]
    idx += 2
    # print(f"after partner facing: idx={idx}")

    # ── partner relative position (2) ────────────────────────────────────
    # signed offset normalized by grid size — layout agnostic
    feat[idx]   = (ox - ax) / (grid_w - 1)
    feat[idx+1] = (oy - ay) / (grid_h - 1)
    idx += 2
    # print(f"after partner relative position: idx={idx}")

    # ── partner reachable (1) ────────────────────────────────────────────
    # 1.0 if we share the same floor region, 0.0 if separated (forced coord)
    feat[idx] = float((ox, oy) in self_reachable)
    idx += 1
    # print(f"after partner reachable: idx={idx}")

    # ── partner adjacency flags (20) ─────────────────────────────────────
    feat[idx:idx+20] = _adjacency_flags((ox, oy), stations, wall_map)
    idx += 20
    # print(f"after partner adjacency: idx={idx}")

    # ── partner previous action (6) ──────────────────────────────────────
    # one-hot over 6 actions — tells agent what partner just did
    feat[idx + prev_action] = 1.0
    idx += 6
    # print(f"after partner previous action: idx={idx}")

    # ── proximity features (21) ──────────────────────────────────────────
    # each target: (dx, dy, reachable) normalized, sentinel (-2,-2,0) if absent

    # static targets — always in the same place per layout
    proximity_targets = [
        stations["pots"],
        stations["onion_dispensers"],
        stations["plate_dispensers"],
        stations["serving_counters"],
    ]

    # dynamic targets — objects currently sitting on counters
    onions_on_counter = [
        tuple(o["position"])
        for o in obs_dict.get("objects", [])
        if o["name"] == "onion"
    ]
    dishes_on_counter = [
        tuple(o["position"])
        for o in obs_dict.get("objects", [])
        if o["name"] == "dish"
    ]
    soups_on_counter = [
        tuple(o["position"])
        for o in obs_dict.get("objects", [])
        if o["name"] == "soup" and (
            "state" not in o
            or (isinstance(o.get("state"), list)
                and len(o["state"]) >= 3
                and o["state"][2] >= cook_time)
        )
    ]

    proximity_targets += [onions_on_counter, dishes_on_counter, soups_on_counter]

    for targets in proximity_targets:
        dx, dy, reachable = _nearest(
            (ax, ay), list(targets), self_reachable, grid_w, grid_h
        )
        feat[idx]   = dx
        feat[idx+1] = dy
        feat[idx+2] = reachable
        idx += 3

    # ── pot states (8) ───────────────────────────────────────────────────
    # 2 pots x (onion_count, cook_progress, is_ready, reachable)
    # pots sorted by position for stable ordering across timesteps
    
    sorted_pots = sorted(stations["pots"], key=lambda p: (p[1], p[0]))
    
    # find matching world object for each pot position
    pot_objects = {
        tuple(o["position"]): o
        for o in obs_dict.get("objects", [])
        if o["name"] == "soup"
        and isinstance(o.get("state"), list)
        and len(o["state"]) >= 3
        and isinstance(o["state"][1], (int, float))
    }
    
    for slot in range(2):
        if slot >= len(sorted_pots):
            # pad with zeros for missing pot slot
            idx += 4
            continue
        pot_pos = sorted_pots[slot]
        obj = pot_objects.get(pot_pos)
        
        if obj is None:
            # pot is empty
            feat[idx]   = 0.0  # onion count
            feat[idx+1] = 0.0  # cook progress
            feat[idx+2] = 0.0  # is ready
        else:
            state = obj["state"]
            n_onions   = int(state[1])
            cook_tick  = int(state[2])
            feat[idx]   = n_onions / 3.0
            feat[idx+1] = cook_tick / cook_time
            feat[idx+2] = float(cook_tick >= cook_time)
        
        # is this pot reachable by self?
        neighbors = [
            (pot_pos[0]+1, pot_pos[1]), (pot_pos[0]-1, pot_pos[1]),
            (pot_pos[0], pot_pos[1]+1), (pot_pos[0], pot_pos[1]-1)
        ]
        feat[idx+3] = float(any(n in self_reachable for n in neighbors))
        idx += 4

    # ── objects on counters (3) ──────────────────────────────────────────
    # raw counts — naturally zero when none present, no sentinel needed
    
    n_onions_on_counter = sum(
        1 for o in obs_dict.get("objects", [])
        if o["name"] == "onion"
    )
    n_dishes_on_counter = sum(
        1 for o in obs_dict.get("objects", [])
        if o["name"] == "dish"
    )
    n_soups_on_counter = sum(
        1 for o in obs_dict.get("objects", [])
        if o["name"] == "soup" and (
            "state" not in o
            or (isinstance(o.get("state"), list)
                and len(o["state"]) >= 3
                and o["state"][2] >= cook_time)
        )
    )
    
    feat[idx]   = float(n_onions_on_counter)
    feat[idx+1] = float(n_dishes_on_counter)
    feat[idx+2] = float(n_soups_on_counter)
    idx += 3
    
    # sanity check — catch any block sizing mistakes during development
    assert idx == FEATURE_DIM, f"Feature index mismatch: got {idx}, expected {FEATURE_DIM}"
    
    return feat

def featurize_jaxmarl(
    obs:        np.ndarray,
    state,
    player_idx: int,
    stations:   Dict[str, set],
    wall_map:   np.ndarray,
    grid_w:     int,
    grid_h:     int,
    cook_time:  int = 20,
    prev_action: int = 4,
) -> np.ndarray:
    """
    Convert a JaxMARL observation and state to an 85-dim feature vector.
    Produces the same output as featurize_symbolic() for the same game state.

    Parameters
    ----------
    obs        : np.ndarray of shape (H, W, 26) — full grid observation
    state      : JaxMARL OvercookedState
    player_idx : which agent we are featurizing for (0 or 1)
    stations   : output of _get_layout_stations()
    wall_map   : 2D boolean numpy array from state.wall_map
    grid_w/h   : grid dimensions
    cook_time  : steps for a pot to finish cooking
    prev_action: action taken at previous timestep (default stay)
    """
    obs  = np.array(obs, dtype=np.float32)
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # agent positions — state.agent_pos[i] is (x, y) = (col, row)
    ax, ay = int(state.agent_pos[player_idx][0]), int(state.agent_pos[player_idx][1])
    ox, oy = int(state.agent_pos[1 - player_idx][0]), int(state.agent_pos[1 - player_idx][1])

    # flood fill from each agent's current position
    self_reachable  = _flood_fill((ax, ay), wall_map)
    other_reachable = _flood_fill((ox, oy), wall_map)

    idx = 0

    # ── self held object (4) ─────────────────────────────────────────────
    feat[idx:idx+4] = _held_onehot_from_inv(state.agent_inv[player_idx])
    idx += 4
    # print(f"after self held: idx={idx}")

    # ── self facing direction (2) ────────────────────────────────────────
    # state.agent_dir[i] is (dx, dy) directly in our convention
    feat[idx]   = int(state.agent_dir[player_idx][0])
    feat[idx+1] = int(state.agent_dir[player_idx][1])
    idx += 2
    # print(f"after self facing: idx={idx}")

    # ── self adjacency flags (20) ────────────────────────────────────────
    feat[idx:idx+20] = _adjacency_flags((ax, ay), stations, wall_map)
    idx += 20
    # print(f"after self adjacency: idx={idx}")

    # ── partner held object (4) ──────────────────────────────────────────
    feat[idx:idx+4] = _held_onehot_from_inv(state.agent_inv[1 - player_idx])
    idx += 4
    # print(f"after partner held: idx={idx}")

    # ── partner facing direction (2) ─────────────────────────────────────
    feat[idx]   = int(state.agent_dir[1 - player_idx][0])
    feat[idx+1] = int(state.agent_dir[1 - player_idx][1])
    idx += 2
    # print(f"after partner facing: idx={idx}")

    # ── partner relative position (2) ────────────────────────────────────
    feat[idx]   = (ox - ax) / (grid_w - 1)
    feat[idx+1] = (oy - ay) / (grid_h - 1)
    idx += 2
    # print(f"after partner position: idx={idx}")

    # ── partner reachable (1) ────────────────────────────────────────────
    feat[idx] = float((ox, oy) in self_reachable)
    idx += 1
    # print(f"after partner reachable: idx={idx}")

    # ── partner adjacency flags (20) ─────────────────────────────────────
    feat[idx:idx+20] = _adjacency_flags((ox, oy), stations, wall_map)
    idx += 20
    # print(f"after partner adjacency: idx={idx}")

    # ── partner previous action (6) ──────────────────────────────────────
    feat[idx + prev_action] = 1.0
    idx += 6
    # print(f"after partner actions: idx={idx}")

    # ── proximity features (21) ──────────────────────────────────────────

    # static targets — from layout stations
    proximity_targets = [
        stations["pots"],
        stations["onion_dispensers"],
        stations["plate_dispensers"],
        stations["serving_counters"],
    ]

    # dynamic targets — items on counters read from observation channels
    # exclude both agent positions to avoid false positives from held items
    agent_positions = {(ax, ay), (ox, oy)}

    def _active_positions(channel: int) -> list:
        """Return global (x, y) positions where a channel is active,
        excluding agent cells."""
        ys, xs = np.where(obs[:, :, channel] > 0.5)
        return [
            (int(x), int(y))
            for y, x in zip(ys, xs)
            if (int(x), int(y)) not in agent_positions
        ]

    onions_on_counter = _active_positions(23)
    plates_on_counter = _active_positions(22)

    # ready soup on counter — ch21 is active at done pots AND plated soups
    # on counters, so exclude pot positions
    soups_on_counter = [
        pos for pos in _active_positions(21)
        if pos not in stations["pots"]
    ]

    proximity_targets += [onions_on_counter, plates_on_counter, soups_on_counter]

    for targets in proximity_targets:
        dx, dy, reachable = _nearest(
            (ax, ay), list(targets), self_reachable, grid_w, grid_h
        )
        feat[idx]   = dx
        feat[idx+1] = dy
        feat[idx+2] = reachable
        idx += 3
    # print(f"after proximity: idx={idx}")

    # ── pot states (8) ───────────────────────────────────────────────────
    # 2 pots x (onion_count, cook_progress, is_ready, reachable)
    # sorted by position for stable ordering — matches featurize_symbolic

    sorted_pots = sorted(stations["pots"], key=lambda p: (p[1], p[0]))

    for slot in range(2):
        if slot >= len(sorted_pots):
            # pad with zeros for missing pot slot
            idx += 4
            continue
        pot_pos = sorted_pots[slot]        
        px, py = pot_pos
        base = idx

        # onion count — ch16 gives count directly (0, 1, 2, or 3)
        onion_count = float(obs[py, px, 16])
        feat[base]   = onion_count / 3.0

        # cook progress and is_ready
        # ch20 active = currently cooking (not yet done)
        # ch21 active = done and ready to scoop
        is_cooking = float(obs[py, px, 20] > 0.5)
        is_ready   = float(obs[py, px, 21] > 0.5)

        # ch18 gives onions in soup (0 or 3) — proxy for cook progress
        # when cooking: increases from 0 to 3 as it approaches done
        # we normalize by cook_time to match featurize_symbolic's convention
        soup_val = float(obs[py, px, 18])
        cook_progress = is_cooking * (soup_val / 3.0)

        feat[base+1] = cook_progress
        feat[base+2] = is_ready

        # reachable — any neighbor of pot in self_reachable
        neighbors = [
            (px+1, py), (px-1, py),
            (px, py+1), (px, py-1)
        ]
        feat[base+3] = float(any(n in self_reachable for n in neighbors))

        idx += 4

    # ── objects on counters (3) ──────────────────────────────────────────
    # reuse the positions we already computed in the proximity block
    feat[idx]   = float(len(onions_on_counter))
    feat[idx+1] = float(len(plates_on_counter))
    feat[idx+2] = float(len(soups_on_counter))
    idx += 3
    # print(f"after objects on counters: idx={idx}")

    assert idx == FEATURE_DIM, \
        f"Feature index mismatch: got {idx}, expected {FEATURE_DIM}"

    return feat