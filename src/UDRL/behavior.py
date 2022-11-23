#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import cast
import logging
import torch
from torch import nn
from torch import Tensor
import gym
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.distributions import make_proba_distribution, Distribution


# TODO: Move shared functionality from UDRLBehaviorCNN to base class
class UDRLBehavior(nn.Module):  # pylint: disable=abstract-method
    # Copied from stable_baselines3/common/policies.py: BaseModel
    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        """
        for param in self.parameters():
            return param.device
        return torch.device("cpu")


class UDRLBehaviorCNN(UDRLBehavior):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, hidden_size: int = 256):
        super().__init__()

        assert not isinstance(action_space, gym.spaces.Box), "Box action space is not supported"

        # Support different action spaces by using ready-to-use stable_baselines3 implementations
        # NOTE: Technically, the action_net layer is supposed to be last, but according to the
        # paper,we need the Bilinear layer last. At least in their CNN experiment, they used it as
        # the last layer. I'm not sure what's the best in this case.
        self.action_dist: Distribution = make_proba_distribution(action_space, use_sde=False)
        self.action_net = cast(nn.Module, self.action_dist.proba_distribution_net(hidden_size))

        # Distribution does not expose the output layer size, so we compute it using a forward pass
        with torch.no_grad():
            action_space_size: int = self.action_net(torch.zeros(hidden_size)).shape[0]
            logging.debug("Action space size: %s", action_space_size)

        self.model = nn.Sequential(
            NatureCNN(observation_space, hidden_size),  # Includes one fc layer + relu
            self.action_net,  # second fc layer
            nn.ReLU(),
        )
        self.bilinear = nn.Bilinear(action_space_size, 2, action_space_size)

    def forward(self, state: Tensor, command: Tensor) -> Tensor:
        """Runs a forward pass. Moves state and command tensors to the corresponding device memory."""
        state = state.to(self.device)
        command = command.to(self.device)

        out = self.model(state)
        out = self.bilinear(out, command)
        out = torch.relu(out)
        return out

    def action(self, state: Tensor, command: Tensor, deterministic: bool = False) -> Tensor:
        """Randomly samples actions based on their predicted probabilities.
        If deterministic is True, it returns the most likely action."""
        probs = self(state, command)
        self.action_dist.proba_distribution(probs)  # pylint: disable=no-value-for-parameter
        return self.action_dist.get_actions(deterministic)
