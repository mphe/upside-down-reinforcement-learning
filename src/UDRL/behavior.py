#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import cast, Tuple
import logging
import torch
from torch import nn
from torch import Tensor
import gym
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.distributions import make_proba_distribution, Distribution
from stable_baselines3.common.preprocessing import preprocess_obs


# TODO: Move all shared functionality from UDRLBehaviorCNN to base class
class UDRLBehavior(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, hidden_size: int) -> None:
        super().__init__()
        self._observation_space: gym.spaces.Space = observation_space

        assert not isinstance(observation_space, gym.spaces.Dict), "Dict observation space not supported"
        assert not isinstance(action_space, gym.spaces.Box), "Box action space is not supported"

        # Support different action spaces by using ready-to-use stable_baselines3 implementations
        # NOTE: Technically, the action_net layer is supposed to be last, but according to the
        # paper,we need the Bilinear layer last. At least in their CNN experiment, they used it as
        # the last layer. I'm not sure what's the best in this case.
        self.action_dist: Distribution = make_proba_distribution(action_space, use_sde=False)
        self.action_net = cast(nn.Module, self.action_dist.proba_distribution_net(hidden_size))

        # Distribution does not expose the output layer size, so we compute it using a forward pass
        with torch.no_grad():
            self._action_space_size: int = self.action_net(torch.zeros(hidden_size)).shape[0]
            logging.debug("Action space size: %s", self._action_space_size)

    def get_action_space_size(self) -> int:
        return self._action_space_size

    def forward(self, _state: Tensor, _command: Tensor) -> Tensor:
        raise NotImplementedError

    def action(self, state: Tensor, command: Tensor, deterministic: bool = False) -> Tensor:
        """Randomly samples actions based on their predicted probabilities.
        If deterministic is True, it returns the most likely action."""
        probs = self(state, command)
        return self._update_dist(probs).get_actions(deterministic)

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Computes the loss according to the action_space used."""
        return self._update_dist(pred).log_prob(target)

    def _update_dist(self, probs: Tensor) -> Distribution:
        return self.action_dist.proba_distribution(probs)  # pylint: disable=no-value-for-parameter

    # Copied from stable_baselines3/common/policies.py: BaseModel
    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.
        """
        for param in self.parameters():
            return param.device
        return torch.device("cpu")

    def _preprocess(self, obs: Tensor, command: Tensor) -> Tuple[Tensor, Tensor]:
        """Preprocesses a tensor and command and moves them to the desired device memory."""
        # We can assume this is not a Dict.
        return (cast(Tensor, preprocess_obs(obs, self._observation_space)).to(self.device),
                command.to(self.device))


class UDRLBehaviorCNN(UDRLBehavior):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, hidden_size: int = 256):
        super().__init__(observation_space, action_space, hidden_size)

        # NatureCNN (copied and adaptedfrom stable baselines 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.model = nn.Sequential(
            # NatureCNN(observation_space, hidden_size),  # Includes one fc layer + relu
            self.cnn,
            nn.Linear(n_flatten, hidden_size),
            nn.ReLU(),
            self.action_net,  # second fc layer
            # nn.ReLU(),
            nn.Tanh()
        )

        self.bilinear = nn.Bilinear(self.get_action_space_size(), 2, self.get_action_space_size())
        # self.bilinear = nn.Bilinear(self.action, 2, hidden_size)

    def forward(self, state: Tensor, command: Tensor) -> Tensor:
        """Runs a forward pass. Moves state and command tensors to the corresponding device memory."""
        state, command = self._preprocess(state, command)
        out = self.model(state)
        # out = self.action_net(out)  # out
        # out = torch.tanh(out)
        out = self.bilinear(out, command)  # second fc
        # out = torch.relu(out)
        return out
