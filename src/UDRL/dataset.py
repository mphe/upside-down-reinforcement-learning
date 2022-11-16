#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from torch import Tensor
from typing import List


class UDRLDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._states: List[Tensor] = []
        self._commands: List[Tensor] = []
        self._actions: List[Tensor] = []

    def add(self, state: Tensor, command: Tensor, action: Tensor) -> None:
        self._states.append(state)
        self._commands.append(command)
        self._actions.append(action)

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int):
        return self._states[idx], self._commands[idx], self._actions[idx]
