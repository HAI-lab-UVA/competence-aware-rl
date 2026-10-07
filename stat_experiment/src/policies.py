from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .models import FactorizedQNetwork


def valid_actions(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask) > 0.5)


def feasible_task_actions(mask: np.ndarray) -> List[int]:
    return [int(a) for a in valid_actions(mask) if int(a) >= 3]


def corrupt_assignment_action_with_info(
    action: int,
    mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[int, bool, bool]:
    """Return action plus whether corruption was possible and applied.

    Corruption is only meaningful when at least two task-selection actions are
    feasible. If the sampled replacement would be identical, keep the original
    action and report applied=False.
    """
    tasks = feasible_task_actions(mask)
    opportunity = len(tasks) >= 2
    if not opportunity or rng.random() >= epsilon:
        return int(action), opportunity, False

    alternatives = [a for a in tasks if a != int(action)]
    if not alternatives:
        return int(action), opportunity, False
    return int(rng.choice(alternatives)), opportunity, True


def corrupt_assignment_action(
    action: int,
    mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    action, _opportunity, _applied = corrupt_assignment_action_with_info(
        action, mask, epsilon, rng
    )
    return action


class PartnerPolicy:
    def act(self, state_dict: dict, agent_idx: int, rng: np.random.Generator) -> int:
        raise NotImplementedError


class NearestTaskPartner(PartnerPolicy):
    """Deterministic masked heuristic used as the initial fixed partner policy."""

    def __init__(self, env):
        self.env = env

    def act(self, state_dict: dict, agent_idx: int, rng: np.random.Generator) -> int:
        mask = np.asarray(state_dict["agent_masks"][agent_idx])
        tasks = feasible_task_actions(mask)
        if not tasks:
            valid = valid_actions(mask)
            return int(valid[0]) if len(valid) else 0

        agent = self.env.model.sorted_agents[agent_idx]
        best_action = tasks[0]
        best_dist = float("inf")
        for action in tasks:
            task = self.env.model.sorted_tasks[action - 3]
            dx = agent.pos[0] - task.pos[0]
            dy = agent.pos[1] - task.pos[1]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_action = action
        return int(best_action)


@dataclass
class FDQNPolicy:
    network: "FactorizedQNetwork"
    device: "torch.device"
    belief_input: Optional[float] = None

    def obs_tensor(self, obs: np.ndarray) -> "torch.Tensor":
        values = np.asarray(obs, dtype=np.float32)
        if self.belief_input is not None:
            values = np.concatenate([values, np.array([self.belief_input], dtype=np.float32)])
        import torch
        return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

    def act(self, state_dict: dict, agent_idx: int, rng: np.random.Generator) -> int:
        import torch
        from .models import masked_argmax

        with torch.no_grad():
            obs = self.obs_tensor(state_dict["obs"])
            q = self.network(obs).squeeze(0)[agent_idx]
            mask = torch.tensor(state_dict["agent_masks"][agent_idx], dtype=torch.float32, device=self.device)
            return masked_argmax(q, mask)


@dataclass
class FDQNPartnerPolicy(PartnerPolicy):
    """Frozen FDQN partner that uses the agent-specific action head."""

    network: "FactorizedQNetwork"
    device: "torch.device"

    def act(self, state_dict: dict, agent_idx: int, rng: np.random.Generator) -> int:
        import torch
        from .models import masked_argmax

        obs = torch.tensor(
            np.asarray(state_dict["obs"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask = torch.tensor(
            state_dict["agent_masks"][agent_idx],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            q = self.network(obs).squeeze(0)[agent_idx]
            return masked_argmax(q, mask)


def load_fdqn_partner_policy(checkpoint_path: str, device: "torch.device") -> FDQNPartnerPolicy:
    import torch
    from .models import FactorizedQNetwork

    payload = torch.load(checkpoint_path, map_location=device)
    net = FactorizedQNetwork(
        payload["input_dim"],
        payload["num_agents"],
        payload["num_actions"],
        hidden_dim=payload["hidden_dim"],
    ).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return FDQNPartnerPolicy(net, device)
