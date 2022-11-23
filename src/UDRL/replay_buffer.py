#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from torch import Tensor
from typing import List, NamedTuple


class Trajectory(NamedTuple):
    summed_rewards: float
    states: List[Tensor]
    actions: List[Tensor]
    rewards: List[float]

    @classmethod
    def create(cls, states: List[Tensor], actions: List[Tensor], rewards: List[float]) -> "Trajectory":
        assert len(states) == len(actions) == len(rewards), "Inconsistent trajectory sizes"
        return Trajectory(sum(rewards), states, actions, rewards)

    def __len__(self) -> int:
        return len(self.rewards)


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self._max_size: int = max_size
        self._buffer: List[Trajectory] = []

    def add_trajectories(self, trajectories: List[Trajectory]) -> None:
        """Adds multiple trajectories in one pass. If exceeding maximum size, the worst episodes are removed."""
        self._buffer.extend(trajectories)

        # Sort in descending order, because we often need to pop the smallest element
        self._buffer.sort(key=lambda t: t.summed_rewards, reverse=True)

        # Pop the last few elements to remove the worst episodes
        # NOTE: We could use slices here, but that creates a needless copy of the list.
        while len(self._buffer) > self._max_size:
            self._buffer.pop()

    def add(self, states: List[Tensor], actions: List[Tensor], rewards: List[float]) -> None:
        """Adds a trajectory to the buffer. If exceeding maximum size, the worst episode is removed."""
        self.add_trajectories([ Trajectory.create(states, actions, rewards) ])

    def k_best(self, k: int) -> List[Trajectory]:
        """Returns K episodes with highest total episode rewards in descending order."""
        return self._buffer[:k]

    def sample(self, k: int) -> List[Trajectory]:
        """Returns K randomly selected episodes."""
        return random.choices(self._buffer, k=k)

    def __len__(self) -> int:
        return len(self._buffer)
