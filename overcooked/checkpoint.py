"""
Checkpoint helpers for saving/loading parameter PyTrees as .npz files.
Works with the parameter structure produced by `my_network.init_model`.
"""
from __future__ import annotations
import os
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict


def _flatten_params(params: Any) -> Dict[str, jnp.ndarray]:
    flat = {}
    def _rec(p, prefix: str):
        if isinstance(p, dict):
            for k, v in p.items():
                _rec(v, f"{prefix}{k}/")
        elif isinstance(p, list):
            for i, v in enumerate(p):
                _rec(v, f"{prefix}{i}/")
        elif (isinstance(p, tuple) and len(p) == 2
              and hasattr(p[0], 'shape') and hasattr(p[1], 'shape')):
            W, b = p
            flat[f"{prefix}W"] = W
            flat[f"{prefix}b"] = b
        else:
            # leaf fallback
            flat[prefix.rstrip('/')] = p
    _rec(params, "")
    return flat


def save_params_npz(params: Any, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    arrays = _flatten_params(params)
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_params_npz(path: str, like_params: Any):
    """
    Rebuild a params PyTree using the structure of `like_params` and the
    arrays stored in `path` (npz). This allows configuration to be inferred
    from your current code while values come from the checkpoint.
    """
    data = np.load(path)
    name_to_arr = {k: jnp.array(data[k]) for k in data.files}

    def _rebuild(p, prefix: str):
        if isinstance(p, dict):
            return {k: _rebuild(v, f"{prefix}{k}/") for k, v in p.items()}
        if isinstance(p, list):
            return [_rebuild(v, f"{prefix}{i}/") for i, v in enumerate(p)]
        if (isinstance(p, tuple) and len(p) == 2
                and hasattr(p[0], 'shape') and hasattr(p[1], 'shape')):
            W = name_to_arr[f"{prefix}W"]
            b = name_to_arr[f"{prefix}b"]
            return (W, b)
        return name_to_arr[prefix.rstrip('/')]

    return _rebuild(like_params, "")


def latest_npz(path_or_dir: str) -> str | None:
    """Return the most recent .npz checkpoint under a directory,
    or the path itself if it is a file.
    """
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        return None
    npzs = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith('.npz')]
    if not npzs:
        return None
    npzs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return npzs[0]
