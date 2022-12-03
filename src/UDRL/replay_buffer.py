#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import numpy as np
import random
import logging
import torch
from torch import Tensor
from typing import List, Sequence


# TODO: Maybe don't store states and actions as tensors, since they're needlessly converted back and
# forth between np.array and Tensor during compression/decompression.
class BaseTrajectory:
    @property
    def states(self) -> List[Tensor]:
        raise NotImplementedError

    @property
    def actions(self) -> List[Tensor]:
        raise NotImplementedError

    @property
    def rewards(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def summed_rewards(self) -> float:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def is_compressed(self) -> bool:
        raise NotImplementedError

    def compressed(self) -> "CompressedTrajectory":
        raise NotImplementedError

    def uncompressed(self) -> "Trajectory":
        raise NotImplementedError


class CompressedTrajectory(BaseTrajectory):
    def __init__(self, summed_rewards: float, data: io.BytesIO, length: int):
        self._summed_rewards = summed_rewards
        self._data: io.BytesIO = data
        self._length: int = length

    @property
    def summed_rewards(self) -> float:
        return self._summed_rewards

    @property
    def states(self) -> List[Tensor]:
        self._warn_access("states")
        return self.uncompressed().states

    @property
    def actions(self) -> List[Tensor]:
        self._warn_access("actions")
        return self.uncompressed().actions

    @property
    def rewards(self) -> np.ndarray:
        self._warn_access("rewards")
        return self.uncompressed().rewards

    @staticmethod
    def _warn_access(field: str):
        logging.warning("Accessing `%s` field of a compressed trajectory. This is potentially slow."
                        "Consider decompressing first, then work on the uncompressed trajectory",
                        field)

    def __len__(self) -> int:
        return self._length

    def is_compressed(self) -> bool:
        return True

    def compressed(self) -> "CompressedTrajectory":
        return self

    def uncompressed(self) -> "Trajectory":
        self._data.seek(0)
        data = np.load(self._data)
        states = [ torch.from_numpy(i) for i in data["states"] ]
        actions = [ torch.from_numpy(i) for i in data["actions"] ]
        return Trajectory(states, actions, data["rewards"], self.summed_rewards)


class Trajectory(BaseTrajectory):
    def __init__(self, states: List[Tensor], actions: List[Tensor], rewards: Sequence[float], summed_rewards: float = -1):
        assert len(states) == len(actions) == len(rewards), "Inconsistent trajectory sizes"
        # Make sure to make a copy to prevent accidental modifications
        self._states: List[Tensor] = list(states)
        self._actions: List[Tensor] = list(actions)
        self._rewards: np.ndarray = np.array(rewards)
        self._summed_rewards: float = summed_rewards if summed_rewards > 0 else sum(rewards)

    @property
    def summed_rewards(self) -> float:
        return self._summed_rewards

    @property
    def states(self) -> List[Tensor]:
        return self._states

    @property
    def actions(self) -> List[Tensor]:
        return self._actions

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards

    def is_compressed(self) -> bool:
        return False

    def compressed(self) -> CompressedTrajectory:
        data = io.BytesIO()
        # Prevent warnings due to Tensor objects
        states = [ i.numpy() for i in self.states ]
        actions = [ i.numpy() for i in self.actions ]
        np.savez_compressed(data, states=states, actions=actions, rewards=self.rewards)
        # uncompressed = len(states) * (4 + 4 * np.prod(actions[0].shape) + np.prod(states[0].shape))
        # compressed = len(data.getvalue())
        # print("Compressed:", compressed)
        # print("Uncompressed:", uncompressed)
        # print("Ratio:", compressed / uncompressed)
        return CompressedTrajectory(self.summed_rewards, data, len(self))

    def uncompressed(self) -> "Trajectory":
        return self

    def __len__(self) -> int:  # pylint: disable=invalid-length-returned
        return self.rewards.size


class ReplayBuffer:
    def __init__(self, max_size: int, compress: bool = False) -> None:
        self._max_size: int = max_size
        self._buffer: List[BaseTrajectory] = []
        self._compress = compress

    def add_trajectories(self, trajectories: List[BaseTrajectory]) -> None:
        """Adds multiple trajectories in one pass. If exceeding maximum size, the worst episodes are removed."""
        if self._compress:
            self._buffer.extend(( t.compressed() for t in trajectories ))
        else:
            self._buffer.extend(trajectories)

        # Sort in descending order, because we often need to pop the smallest element
        self._buffer.sort(key=lambda t: t.summed_rewards, reverse=True)

        # Pop the last few elements to remove the worst episodes
        # NOTE: We could use slices here, but that creates a needless copy of the list.
        while len(self._buffer) > self._max_size:
            self._buffer.pop()

    def add(self, states: List[Tensor], actions: List[Tensor], rewards: Sequence[float]) -> None:
        """Adds a trajectory to the buffer. If exceeding maximum size, the worst episode is removed."""
        self.add_trajectories([ Trajectory(states, actions, rewards) ])

    def k_best(self, k: int) -> List[BaseTrajectory]:
        """Returns K episodes with highest total episode rewards in descending order."""
        return self._buffer[:k]

    def sample(self, k: int) -> List[BaseTrajectory]:
        """Returns K randomly selected episodes."""
        return random.choices(self._buffer, k=k)

    def __len__(self) -> int:
        return len(self._buffer)
