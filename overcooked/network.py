"""
network.py — MLP policy/value network in pure JAX.
Shared trunk -> policy head (logits) + value head (scalar).
Compatible with checkpoint.py's save/load format.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp

Params = Dict


def _glorot(key: jax.Array, in_dim: int, out_dim: int,
            scale: float = 1.0) -> Tuple:
    lim = scale * math.sqrt(6.0 / (in_dim + out_dim))
    W = jax.random.uniform(key, (out_dim, in_dim), minval=-lim, maxval=lim)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return (W, b)


def init_model(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    key: Optional[jax.Array] = None,
) -> Params:
    """
    Initialise MLP parameters.

    Returns a dict:
        {
          'layers': [(W, b), ...],   # trunk hidden layers
          'pi':     (W, b),          # policy head
          'v':      (W, b),          # value head
        }
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    keys = jax.random.split(key, len(hidden_sizes) + 2)
    layers: List = []
    in_dim = obs_dim

    for i, h in enumerate(hidden_sizes):
        layers.append(_glorot(keys[i], in_dim, h))
        in_dim = h

    return {
        "layers": layers,
        "pi": _glorot(keys[-2], in_dim, act_dim, scale=0.01),
        "v":  _glorot(keys[-1], in_dim, 1,       scale=0.01),
    }


def forward(params: Params, x: jnp.ndarray):
    """
    Forward pass.

    Returns
    -------
    pi_logits : (..., act_dim)
    value     : (...,)
    """
    h = x
    for (W, b) in params["layers"]:
        h = jax.nn.relu(h @ W.T + b)

    pi = h @ params["pi"][0].T + params["pi"][1]
    v  = (h @ params["v"][0].T + params["v"][1]).squeeze(-1)
    return pi, v