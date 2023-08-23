#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import cast, Tuple, List
import logging
import torch
from torch import nn
from torch import Tensor
import gym
from stable_baselines3.common.torch_layers import FlattenExtractor, create_mlp
from stable_baselines3.common.distributions import make_proba_distribution, Distribution
from stable_baselines3.common.preprocessing import preprocess_obs


def conv(*args, **kwargs):
    c = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(c.weight, nonlinearity="relu")
    nn.init.constant_(c.bias, 0)
    return c


def linear(*args, **kwargs):
    fc = nn.Linear(*args, **kwargs)
    nn.init.xavier_normal_(fc.weight, gain=nn.init.calculate_gain("relu"))
    nn.init.constant_(fc.bias, 0)
    return fc


class UDRLBehavior(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, hidden_size: int) -> None:
        super().__init__()
        self._observation_space: gym.spaces.Space = observation_space

        assert not isinstance(observation_space, gym.spaces.Dict), "Dict observation space not supported"
        assert not isinstance(action_space, gym.spaces.Box), "Box action space is not supported"

        # Support different action spaces by using ready-to-use stable_baselines3 implementations
        # NOTE: Technically, the action_net layer is supposed to be last, but according to the
        # UDRL paper, the MI layer is last, at least in their CNN experiment.
        # I'm not sure what's the best in this case.
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


class MIGatedLayer(nn.Module):
    """Gated multiplicative interaction layer. See section 'C -  Network Architectures' in the paper."""
    def __init__(self, in_size_1: int, in_size_2: int, out_size: int, in_2_activ_fn=nn.ReLU) -> None:
        super().__init__()

        self._in1 = nn.Sequential(
            linear(in_size_1, out_size),
            nn.Sigmoid()
        )

        if in_2_activ_fn:
            self._in2 = nn.Sequential(
                linear(in_size_2, out_size),
                in_2_activ_fn()
            )
        else:
            self._in2 = linear(in_size_2, out_size)

    def forward(self, in1: Tensor, in2: Tensor) -> Tensor:
        g = self._in1(in1)
        x = self._in2(in2)
        return g * x


class UDRLBehaviorMLP(UDRLBehavior):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 net_arch: List[int] = [128, 128], activ_fn=nn.Tanh):
        super().__init__(observation_space, action_space, net_arch[-1])
        self._flatten = FlattenExtractor(observation_space)

        # NOTE: The paper is not clear about whether the MI layer counts as the first linear layer
        # or is an extra layer inserted before the MLP part.
        # Here, we use it as the first linear layer, hence we reduce the net_arch correspondingly.
        self._gated = MIGatedLayer(2, self._flatten.features_dim, net_arch[0])

        # output_dim = 0, so it doesn't add an extra output layer. We handle that separately.
        self._mlp = nn.Sequential(*create_mlp(net_arch[0], 0, net_arch[1:], activ_fn))

    def forward(self, state: Tensor, command: Tensor) -> Tensor:
        state, command = self._preprocess(state, command)
        out = self._flatten(state)
        out = self._gated(command, out)
        out = self._mlp(out)
        out = self.action_net(out)
        return out


class UDRLBehaviorCNN(UDRLBehavior):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, hidden_size: int = 256):
        super().__init__(observation_space, action_space, hidden_size)

        # NatureCNN (copied and adapted from stable baselines 3)
        self.cnn = nn.Sequential(
            conv(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            conv(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            conv(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # self._gated = MIGatedLayer(2, self.get_action_space_size(), self.get_action_space_size())
        self._gated = MIGatedLayer(2, hidden_size, self.get_action_space_size(), None)

        self.model = nn.Sequential(
            self.cnn,
            linear(n_flatten, hidden_size),
            nn.ReLU(),
            linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Tanh(),
            # self.action_net,  # second fc layer
            # nn.ReLU(),
            # nn.Tanh()
        )

    def forward(self, state: Tensor, command: Tensor) -> Tensor:
        """Runs a forward pass. Moves state and command tensors to the corresponding device memory."""
        state, command = self._preprocess(state, command)
        out = self.model(state)
        out = self._gated(command, out)
        return out
