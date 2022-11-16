#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Original source: https://github.com/BY571/Upside-Down-Reinforcement-Learning/blob/f27215bb6bc487d6f587706432373e88f6a07091/Upside-Down.ipynb

import logging
from typing import Any, Dict, Optional, List, Union, Callable, Tuple
import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
# from stable_baselines3.common.base_class import BaseAlgorithm
import copy
import tqdm

from .behavior import UDRLBehaviorCNN
from .replay_buffer import ReplayBuffer
from .dataset import UDRLDataset

# mypy complains about np.floating not being compatible to basic python floats
Float = Union[float, np.floating]

EnvStepCallback = Callable[[Dict[str, Any], Dict[str, Any]], Any]


@dataclass
class Command:
    reward: Float
    horizon: int

    def update(self, collected_reward, max_reward: Optional[float] = None) -> None:
        # TODO: The desired reward becomes negative when it reaches the desired horizon
        # without the episode being finished, yet. In further steps, it will try to
        # accumulate more and more negative reward until the episode ends.
        # This doesn't sound right, but the paper didn't mention anything about it.
        self.reward -= collected_reward
        if max_reward is not None:
            self.reward = min(self.reward, max_reward)
        self.horizon = max(self.horizon - 1, 1)


@dataclass
class TrainStats:
    reward_history: List[float]
    loss_history: List[Float]
    average_100_reward: List[Float]
    command_history: List[Command]

    def plot(self) -> None:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        plt.title("Rewards")
        plt.plot(self.reward_history, label="rewards")
        plt.plot(self.average_100_reward, label="average100")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.title("Loss")
        plt.plot(self.loss_history)
        plt.subplot(2, 2, 3)
        plt.title("desired Rewards")
        plt.plot([ i.reward for i in self.command_history ])
        plt.subplot(2, 2, 4)
        plt.title("desired Horizon")
        plt.plot([ i.horizon for i in self.command_history ])
        plt.show()


class UDRL:
    def __init__(self,
                 env: gym.Env,
                 horizon_scale: float = 0.02,
                 return_scale: float = 0.02,
                 replay_size: int = 700,
                 n_warm_up_episodes: int = 50,
                 n_updates_per_iter: int = 100,
                 n_episodes_per_iter: int = 15,
                 last_few: int = 50,
                 batch_size: int = 256,
                 max_reward: Optional[float] = None,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        self._horizon_scale = horizon_scale
        self._return_scale = return_scale
        self._replay_size = replay_size
        self._n_warm_up_episodes = n_warm_up_episodes
        self._n_updates_per_iter = n_updates_per_iter
        self._n_episodes_per_iter = n_episodes_per_iter
        self._last_few = last_few
        self._batch_size = batch_size
        self._max_reward = max_reward

        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)

        self._env = env
        self._obs_space: gym.spaces.Space = VecTransposeImage.transpose_space(self._env.observation_space)
        logging.debug("Observation space: %s", self._obs_space.shape)

        # TODO: Support vectorized environments and wrap environment
        # self._env = BaseAlgorithm._wrap_env(self._env, verbose=1)

        policy_kwargs = {} if policy_kwargs is None else {}
        self._behavior = UDRLBehaviorCNN(self._obs_space, env.action_space, **policy_kwargs).to(self._device)
        self._optimizer = optim.Adam(params=self._behavior.parameters())
        self._replay_buffer = ReplayBuffer(self._replay_size)

        self._evaluation_command: Command = Command(1.0, 1)

    def learn(self, max_iterations: int, skip_warmup: bool = False, env_step_callback: Optional[EnvStepCallback] = None) -> TrainStats:
        if not skip_warmup:
            logging.info("Warm up for %s episodes...", self._n_warm_up_episodes)
            for _ in tqdm.trange(self._n_warm_up_episodes):
                self._generate_episode(use_bf=False, add_to_replay_buffer=True,
                                       env_step_callback=env_step_callback)

        return self._run_upside_down(max_iterations, env_step_callback)

    def action(self, obs: np.ndarray, command: Command, deterministic: bool = False) -> np.ndarray:
        """Randomly samples actions based on their predicted probabilities for the given observation
        and command. If deterministic is True, it returns the most likely action."""
        return self._action(self._preprocess_state_to_tensor(obs),
                            self._create_command_tensor(command),
                            deterministic).cpu().numpy()

    def save(self, fname: str) -> None:
        torch.save(self._behavior.state_dict(), fname)

    def load(self, fname: str) -> None:
        self._behavior.load_state_dict(torch.load(fname))

    def _build_train_dataset(self, num: int) -> UDRLDataset:
        ds = UDRLDataset()
        episodes = self._replay_buffer.sample(num)

        for ep in episodes:
            # For episodic tasks only trailing segments are considered. See section 2.2.3.
            eplength: int = len(ep)
            t1: int = np.random.randint(0, eplength - 1)
            t2: int = eplength

            state = ep.states[t1]
            command = Command(reward=sum(ep.rewards[t1:t2]), horizon=t2 - t1)
            action = ep.actions[t1]

            ds.add(state, self._create_command_tensor(command), action)

        return ds

    def _train_behavior_function(self) -> Float:
        ds = self._build_train_dataset(self._batch_size * self._n_updates_per_iter)
        loader = DataLoader(ds, batch_size=self._batch_size, shuffle=False)
        losses: List[float] = []

        with tqdm.tqdm(total=self._n_updates_per_iter) as progress:
            for states, commands, actions in loader:
                states = states.to(self._device)
                commands = commands.to(self._device)
                actions = actions.to(self._device)

                y_pred = self._behavior(states, commands)  # pylint is drunk  # pylint: disable=not-callable
                self._optimizer.zero_grad(set_to_none=True)
                pred_loss = F.cross_entropy(y_pred, actions)
                pred_loss.backward()
                self._optimizer.step()

                losses.append(pred_loss.item())
                progress.update()

        return np.mean(losses)

    def _sample_exploratory_command(self) -> Command:
        """Returns a tuple (desired_reward, desired_horizon) for exploration. See paper section 2.2.4."""
        k_best = self._replay_buffer.k_best(self._last_few)

        returns: np.ndarray = np.array([ i.summed_rewards for i in k_best ])
        mean_returns = returns.mean()
        std_returns = returns.std()

        new_desired_reward = np.random.uniform(mean_returns, mean_returns + std_returns)
        new_desired_horizon = int(np.mean([ len(i) for i in k_best ]))

        # For evaluation, a command derived from the most recent exploratory command is used.
        # See section 2.2.6.
        self._evaluation_command = Command(mean_returns, new_desired_horizon)

        return Command(new_desired_reward, new_desired_horizon)

    def _generate_episode(self, use_bf: bool, add_to_replay_buffer: bool,
                          command: Optional[Command] = None,
                          env_step_callback: Optional[EnvStepCallback] = None) -> float:
        """Plays an episode using random actions or the behavior function and adds it to the replay buffer."""
        # NOTE: This function got a bit large and a little complicated, but its the only reasonable
        # way to not have high amounts of code duplication that I can think of at the moment.
        states: List[Tensor] = []
        actions: List[Tensor] = []
        rewards: List[float] = []

        state: Tensor = self._preprocess_state_to_tensor(self._env.reset())
        action: Tensor
        summed_reward: float = 0.0

        if use_bf and command is None:
            command = self._sample_exploratory_command()

        while True:
            if use_bf:
                with torch.no_grad():
                    action = self._action(state, self._create_command_tensor(command))
            else:
                action = torch.tensor(self._env.action_space.sample())

            action = action.cpu().float()

            next_state, reward, done, _ = self._env.step(action.numpy())
            summed_reward += reward

            if env_step_callback is not None:
                env_step_callback(locals(), globals())

            if add_to_replay_buffer:
                states.append(state)
                actions.append(action)
                rewards.append(reward)

            state = self._preprocess_state_to_tensor(next_state)

            if use_bf:
                command.update(reward, self._max_reward)

            if done:
                break

        if add_to_replay_buffer:
            self._replay_buffer.add(states, actions, rewards)

        return summed_reward

    def _run_upside_down(self, max_iterations: int, env_step_callback: Optional[EnvStepCallback] = None) -> TrainStats:
        """Algorithm 1 - Upside-Down Reinforcement Learning"""
        stats = TrainStats([], [], [], [])

        for train_iter in range(1, max_iterations + 1):
            logging.info("Training...")
            bf_loss = self._train_behavior_function()
            stats.loss_history.append(bf_loss)

            logging.info("Exploring %s episodes...", self._n_episodes_per_iter)
            for _ in tqdm.trange(self._n_episodes_per_iter):
                self._generate_episode(True, True, env_step_callback=env_step_callback)

            logging.info("Evaluating...")
            stats.command_history.append(self._evaluation_command)
            ep_rewards = self._generate_episode(True, False, command=self._evaluation_command,
                                                env_step_callback=env_step_callback)
            stats.reward_history.append(ep_rewards)
            average_100_reward = np.mean(stats.reward_history[-100:])
            stats.average_100_reward.append(average_100_reward)

            # pylint: disable=logging-fstring-interpolation
            logging.info(f"""
|--------------------------------------------------
| Iteration: {train_iter}/{max_iterations}
| Evaluated reward: {ep_rewards:.2f}
| Mean 100 evaluated rewards: {average_100_reward:.2f}
| Loss: {bf_loss:.2f}
|--------------------------------------------------""")

        return stats

    def _action(self, obs: Tensor, command: Tensor, deterministic: bool = False) -> Tensor:
        """Same as self.action() but works with Tensors and returns a tensor (in CPU memory)."""
        # The input requires a batch dimension, so we add it using [None] indexing.
        return self._behavior.action(obs[None], command[None], deterministic).cpu().squeeze().float()

    def _create_command_tensor(self, command: Command) -> Tensor:
        """Creates a command tensor."""
        return torch.tensor((command.reward * self._return_scale,
                             command.horizon * self._horizon_scale)).float()

    def _preprocess_state_to_tensor(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tensor:
        """Preprocess and convert an observation to torch.Tensor."""
        t, _ = self._obs_to_tensor(obs)
        t = preprocess_obs(t, self._obs_space)
        return t

    # Copied from stable_baselines3/common/policies.py: BaseModel
    def _obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[Tensor, bool]:
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
                obs_space = self._obs_space.spaces[key]  # pylint: disable=no-member
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self._obs_space[key].shape)  # pylint: disable=unsubscriptable-object

        elif is_image_space(self._obs_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self._obs_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self._obs_space)
            # Add batch dimension if needed
            # TODO: support vectorized environments
            # observation = observation.reshape((-1,) + self._obs_space.shape)

        observation = obs_as_tensor(observation, self._device)
        return observation, vectorized_env  # type: ignore[return-value]
