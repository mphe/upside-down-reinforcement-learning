#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Original source: https://github.com/BY571/Upside-Down-Reinforcement-Learning/blob/f27215bb6bc487d6f587706432373e88f6a07091/Upside-Down.ipynb

from torch import Tensor
import random
import numpy as np
from typing import List, NamedTuple


class Trajectory(NamedTuple):
    summed_rewards: float
    states: List[Tensor]
    actions: List[Tensor]
    rewards: List[float]

    def __len__(self) -> int:
        return len(self.rewards)


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self._max_size: int = max_size
        self._buffer: List[Trajectory] = []

    def add(self, states: List[Tensor], actions: List[Tensor], rewards: List[float]) -> None:
        """Adds a trajectory to the buffer. If exceeding maximum size, the worst episode is removed."""
        # The list is kept sorted, hence we can simply pop the last element to remove the worst episode
        if len(self._buffer) == self._max_size:
            self._buffer.pop()

        self._buffer.append(Trajectory(np.sum(rewards), states, actions, rewards))

        # Sort in descending order, because we often need to pop the smallest element
        self._buffer.sort(key=lambda t: t.summed_rewards, reverse=True)

    def k_best(self, k: int) -> List[Trajectory]:
        """Returns K episodes with highest total episode rewards in descending order."""
        return self._buffer[:k]

    def sample(self, k: int) -> List[Trajectory]:
        """Returns K randomly selected episodes."""
        return random.choices(self._buffer, k=k)

    def __len__(self) -> int:
        return len(self._buffer)
