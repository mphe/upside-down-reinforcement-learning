#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Union, Tuple
import copy
import numpy as np
from torch import Tensor
import gym
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor


class Preprocessor:
    def __init__(self, observation_space: gym.spaces.Space):
        self._observation_space: gym.spaces.Space = observation_space

    def transform(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tensor:
        """Preprocess and convert an observation to torch.Tensor."""
        t, _ = self._obs_to_tensor(obs)
        t = preprocess_obs(t, self._observation_space)  # type: ignore[assignment]  # preprocess_obs() has wrong type hints
        return t

    # Copied and minmally adapted from stable_baselines3/common/policies.py: BaseModel
    def _obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]],
                       add_batch_dimension: bool = False) -> Tuple[Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self._observation_space.spaces[key]  # pylint: disable=no-member
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                if add_batch_dimension:
                    observation[key] = obs_.reshape((-1,) + self._observation_space[key].shape)  # pylint: disable=unsubscriptable-object

        elif is_image_space(self._observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self._observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self._observation_space)
            # Add batch dimension if needed
            if add_batch_dimension:
                observation = observation.reshape((-1,) + self._observation_space.shape)

        # NOTE: In our workflow we don't want to move the tensor to the GPU right away when this
        # function gets called..
        obs_tensor = obs_as_tensor(observation, "cpu")  # type: ignore[arg-type]
        return obs_tensor, vectorized_env  # type: ignore[return-value]
