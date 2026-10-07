"""
agent_clones.py — Helpers for loading pre-trained behavioral clones and
their scores, for use as frozen training partners.

competence_labels.json format
-------------------
{
  "clone_name_a": {
    "clone_0": {"competence": 5},
    "clone_1": {"competence": 4}
  },
  ...
}

Each top-level key is a clone checkpoint stem (without .npz). The corresponding
checkpoint is expected at <clones_dir>/<clone_name.npz>.

Scores are split by role because a clone's evaluated scores differ
depending on which seat it occupies during evaluation. The role used for
extra_features lookup always matches the role the clone actually occupies
as a partner in that specific update — see clone_partner_scores().

training_clones entries (in agents.json)
-----------------------------------------
Each element is either:
  - A string  "clone_03.npz"               → train against this clone in
                                              BOTH clone_0 and clone_1
  - A list    ["clone_03.npz", "clone_1"]   → train against this clone in
                                              the specified role only

Use expand_training_clones() to normalise a training_clones list into a
flat list of (clone_name, role) tuples before use.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from network import init_model, forward
from checkpoint import load_params_npz
from bc_features import FEATURE_DIM, _get_layout_stations, featurize_jaxmarl

ACT_DIM   = 6
ALL_ROLES  = ("clone_0", "clone_1")


# ── training_clones parsing ───────────────────────────────────────────────────

def expand_training_clones(
    training_clones: List,
) -> List[Tuple[str, str]]:
    """
    Normalise a training_clones list from agents.json into a flat list of
    (clone_name, role) tuples.

      "clone_03.npz"              → [("clone_03.npz", "clone_0"),
                                      ("clone_03.npz", "clone_1")]
      ["clone_03.npz", "clone_1"] → [("clone_03.npz", "clone_1")]

    The returned list is the full pool from which per-update partners are
    sampled (with shuffling) by agent_population.py.
    """
    pool: List[Tuple[str, str]] = []
    for entry in training_clones:
        if isinstance(entry, str):
            pool.append((entry, "clone_0"))
            pool.append((entry, "clone_1"))
        elif isinstance(entry, list) and len(entry) == 2:
            clone_name, role = entry
            if role not in ALL_ROLES:
                raise ValueError(
                    f"Invalid role '{role}' in training_clones entry "
                    f"{entry!r} — must be one of {ALL_ROLES}"
                )
            pool.append((clone_name, role))
        else:
            raise ValueError(
                f"training_clones entry {entry!r} must be a string "
                f"(clone filename) or a two-element list "
                f"[clone_filename, role]"
            )
    return pool


def learner_player_idx(clone_role: str) -> int:
    """
    Return the seat index the LEARNER should occupy given the clone's role.
    Clone clone_0 → clone is in seat 0 → learner is in seat 1.
    Clone clone_1 → clone is in seat 1 → learner is in seat 0.
    """
    return 1 if clone_role == "clone_0" else 0


def clone_player_idx(clone_role: str) -> int:
    """Return the seat index the CLONE occupies given its role."""
    return 0 if clone_role == "clone_0" else 1


# ── score helpers ─────────────────────────────────────────────────────────────

def _scores_key(clone_name: str) -> str:
    """
    Return the key used to look up a clone in artifacts/partners/competence_labels.json.
    agents.json training_clones entries include the .npz extension
    (since that's the actual filename), but artifacts/partners/competence_labels.json keys do not.
    """
    return clone_name[:-4] if clone_name.endswith(".npz") else clone_name


def load_clone_scores(clones_json_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load artifacts/partners/competence_labels.json — a dict of:

        clone_name (no .npz) -> {"clone_0": {score_name: value, ...},
                                  "clone_1": {score_name: value, ...}}
    """
    with open(clones_json_path) as f:
        return json.load(f)


def clone_partner_scores(
    clone_scores:   Dict[str, Dict[str, Dict[str, float]]],
    clone_name:     str,
    clone_role:     str,
    extra_features: List[str],
) -> List[float]:
    """
    Return an ordered list of score values for the given clone and the
    role it is currently occupying as a partner, one per entry in
    extra_features (in list order).

    clone_name may include the .npz extension — it is stripped
    automatically before looking up artifacts/partners/competence_labels.json.

    Returns an empty list when extra_features is empty.
    """
    if not extra_features:
        return []
    key = _scores_key(clone_name)
    role_scores = clone_scores[key][clone_role]
    return [float(role_scores[feat]) for feat in extra_features]


def clone_checkpoint_path(clones_dir: str, clone_name: str) -> str:
    """
    Resolve the .npz checkpoint path for a clone by filename.
    clone_name is expected to already include the .npz extension.
    """
    return os.path.join(clones_dir, clone_name)


# ── validation ────────────────────────────────────────────────────────────────

def validate_clones(
    training_clones: List,
    clones_dir:      str,
    clone_scores:    Dict[str, Dict[str, Dict[str, float]]],
    extra_features:  List[str] = None,
) -> None:
    """
    Sanity-check that every clone referenced in training_clones has:
      - a checkpoint on disk at clones_dir/<clone_name>
      - an entry in artifacts/partners/competence_labels.json
      - scores for the relevant role(s) under that entry
      - scores for every key in extra_features under those roles

    String entries are checked against both clone_0 and clone_1.
    List entries are checked against the specified role only.

    Raises ValueError with a descriptive message on the first problem.
    """
    if extra_features is None:
        extra_features = []

    if not training_clones:
        raise ValueError(
            "training_clones is empty — at least one clone is required"
        )

    pool = expand_training_clones(training_clones)

    for clone_name, role in pool:
        ckpt = clone_checkpoint_path(clones_dir, clone_name)
        if not os.path.exists(ckpt):
            raise ValueError(
                f"clone '{clone_name}' has no checkpoint at {ckpt}"
            )

        if extra_features:
            key = _scores_key(clone_name)
            if key not in clone_scores:
                raise ValueError(
                    f"clone '{clone_name}' (key '{key}') is missing from "
                    f"artifacts/partners/competence_labels.json, but extra_features={extra_features} "
                    f"requires a score entry for every training clone"
                )
            if role not in clone_scores[key]:
                raise ValueError(
                    f"clone '{clone_name}' is missing '{role}' scores in "
                    f"artifacts/partners/competence_labels.json (available: "
                    f"{list(clone_scores[key].keys())})"
                )
            role_scores = clone_scores[key][role]
            for feat in extra_features:
                if feat not in role_scores:
                    raise ValueError(
                        f"clone '{clone_name}' has no '{feat}' score under "
                        f"'{role}' in artifacts/partners/competence_labels.json (available: "
                        f"{list(role_scores.keys())})"
                    )


# ── clone partner ─────────────────────────────────────────────────────────────

class ClonePartner:
    """
    A frozen, pre-trained clone policy used as a fixed training partner.

    Loads the clone's checkpoint once and exposes make_partner_fn(), which
    builds a partner_fn compatible with ppo_core.collect_rollout. The
    clone's weights never change during training — there is no syncing.

    The clone's seat (player_idx) is specified at make_partner_fn() time
    rather than at construction, since the same checkpoint may be used in
    either seat depending on the role assigned for a given update.
    """

    def __init__(
        self,
        name:         str,
        checkpoint:   str,
        hidden_sizes: tuple = (256, 256),
        seed:         int   = 0,
    ):
        self.name       = name
        self.checkpoint = checkpoint

        like_params = init_model(
            FEATURE_DIM, ACT_DIM,
            hidden_sizes=hidden_sizes,
            key=jax.random.PRNGKey(seed),
        )
        self.params = load_params_npz(checkpoint, like_params=like_params)

    def make_partner_fn(
        self,
        stations:   dict,
        wall_map:   np.ndarray,
        grid_w:     int,
        grid_h:     int,
        player_idx: int,
        cook_time:  int = 20,
    ):
        """
        Build a partner_fn(obs, state, key) -> action closure for this
        clone, suitable for ppo_core.collect_rollout.

        player_idx is the seat this clone occupies:
          clone_0 → player_idx=0, clone_1 → player_idx=1
        Use clone_player_idx(role) to compute the correct value.
        """
        params     = self.params
        _player_idx = player_idx

        def partner_fn(partner_obs, partner_state, partner_key):
            feat = featurize_jaxmarl(
                obs        = np.array(partner_obs),
                state      = partner_state,
                player_idx = _player_idx,
                stations   = stations,
                wall_map   = wall_map,
                grid_w     = grid_w,
                grid_h     = grid_h,
                cook_time  = cook_time,
            )
            logits, _ = forward(params, jnp.asarray(feat))
            return jax.random.categorical(partner_key, logits)

        return partner_fn


def load_clone_partners(
    clone_names:  List[str],
    clones_dir:   str,
    hidden_sizes: tuple = (256, 256),
    seed:         int   = 0,
) -> Dict[str, ClonePartner]:
    """
    Load a ClonePartner for every name in clone_names. Checkpoints are
    loaded once and reused across roles — a single ClonePartner instance
    can be used for either role via make_partner_fn(player_idx=...).
    Call this once per worker process, not once per rollout.
    """
    partners = {}
    for name in clone_names:
        ckpt = clone_checkpoint_path(clones_dir, name)
        partners[name] = ClonePartner(
            name         = name,
            checkpoint   = ckpt,
            hidden_sizes = hidden_sizes,
            seed         = seed,
        )
    return partners