from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedQNetwork(nn.Module):
    """FDQN-style shared trunk with one action head per agent."""

    def __init__(self, input_dim: int, num_agents: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.agent_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_actions) for _ in range(num_agents)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        return torch.stack([head(h) for head in self.agent_heads], dim=1)


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int:
    masked = q_values.clone()
    masked[mask == 0] = -1e8
    return int(torch.argmax(masked).item())
