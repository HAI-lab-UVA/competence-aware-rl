from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


Transition = Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, bool, float]


@dataclass
class ReplayBuffer:
    capacity: int

    def __post_init__(self) -> None:
        self.data: Deque[Transition] = deque(maxlen=self.capacity)

    def append(self, transition: Transition) -> None:
        self.data.append(transition)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        obs, mask, action, reward, next_obs, next_mask, done, belief = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(next_mask, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            np.asarray(belief, dtype=np.float32),
        )
