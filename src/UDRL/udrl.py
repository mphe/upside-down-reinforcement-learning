#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import copy
from typing import Any, Dict, Optional, List, Union, Callable
import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import obs_as_tensor
import tqdm

from .behavior import UDRLBehaviorCNN
from .replay_buffer import ReplayBuffer, Trajectory
from .dataset import UDRLDataset

EnvStepCallback = Callable[[Dict[str, Any], Dict[str, Any]], Any]

# mypy complains about np.floating not being compatible to basic python floats
Float = Union[float, np.floating]


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
            self.reward = min(self.reward, max_reward)  # type: ignore[type-var]  # mypy is drunk
        self.horizon = max(self.horizon - 1, 1)

    def duplicate(self) -> "Command":
        return Command(self.reward, self.horizon)

    def __str__(self) -> str:
        return f"Reward: {self.reward:.2f}, Horizon: {self.horizon}"


@dataclass
class TrainStats:
    reward_history: List[float]
    loss_history: List[Float]
    average_100_reward: List[Float]
    command_history: List[Command]

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({
            "rewards": self.reward_history,
            "loss": self.loss_history,
            "average_100_reward": self.average_100_reward,
            "command_horizon": [ c.horizon for c in self.command_history ],
            "command_reward": [ c.reward for c in self.command_history ],
        })

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
                 env: Union[gym.Env, VecEnv],
                 horizon_scale: float = 0.02,
                 return_scale: float = 0.02,
                 replay_size: int = 700,
                 n_warm_up_episodes: int = 50,
                 n_updates_per_iter: int = 100,
                 n_episodes_per_iter: int = 15,
                 last_few: int = 50,
                 batch_size: int = 256,
                 max_reward: Optional[float] = None,
                 only_trailing_segments: bool = True,
                 compress_replay_buffer: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None,
                 learning_rate: float = 1e-3):
        self._horizon_scale = horizon_scale
        self._return_scale = return_scale
        self._replay_size = replay_size
        self._n_warm_up_episodes = n_warm_up_episodes
        self._n_updates_per_iter = n_updates_per_iter
        self._n_episodes_per_iter = n_episodes_per_iter
        self._last_few = last_few
        self._batch_size = batch_size
        self._max_reward = max_reward
        self._only_trailing_segments = only_trailing_segments

        if not device:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Transform the environment to a vectorized environment and wrap VecTransposeImage for image observations
        self._env: VecEnv = BaseAlgorithm._wrap_env(env, verbose=1)

        obs_space: gym.spaces.Space = self._env.observation_space
        logging.debug("Observation space: %s", obs_space.shape)

        policy_kwargs = {} if policy_kwargs is None else {}
        self._behavior = UDRLBehaviorCNN(obs_space, env.action_space, **policy_kwargs).to(self._device)
        self._optimizer = optim.Adam(params=self._behavior.parameters(), lr=learning_rate)
        self._replay_buffer = ReplayBuffer(self._replay_size, compress=compress_replay_buffer)
        self._training_stats: TrainStats = TrainStats([], [], [], [])

        # This is set in _sample_exploratory_command()
        self._evaluation_command: Command = Command(1.0, 1)

    def get_training_stats(self) -> TrainStats:
        return self._training_stats

    def learn(self, max_iterations: int, skip_warmup: bool = False, env_step_callback: Optional[EnvStepCallback] = None) -> TrainStats:
        if not skip_warmup:
            logging.info("Warm up for %s episodes...", self._n_warm_up_episodes)
            self._generate_episodes(self._n_warm_up_episodes, use_bf=False,
                                    add_to_replay_buffer=True, env_step_callback=env_step_callback)

        return self._run_upside_down(max_iterations, env_step_callback)

    def action(self, obs: np.ndarray, commands: List[Command], deterministic: bool = False) -> np.ndarray:
        """Returns actions based on their predicted probabilities for the given observations and
        commands. If deterministic is True, it returns the most likely actions."""
        return self._behavior.action(self._obs_to_tensor(obs),
                                     self._create_vec_command_tensor(commands),
                                     deterministic).cpu().float().numpy()

    def save(self, fname: str) -> None:
        torch.save(self._behavior.state_dict(), fname)

    def load(self, fname: str) -> None:
        self._behavior.load_state_dict(torch.load(fname))

    def _build_train_dataset(self, num: int) -> UDRLDataset:
        ds = UDRLDataset()
        episodes = self._replay_buffer.sample(num)
        episodes.sort(key=id)  # Sort it by id, so that same trajectories are sorted together
        uncompressed: Trajectory = None  # Store the current uncompressed trajectory
        last_traj = None

        for ep in episodes:
            if last_traj is not ep:
                last_traj = ep
                uncompressed = ep.uncompressed()

            # The paper only considers trailing segments for episodic tasks, see section 2.2.3.
            # We provide an option for that.
            eplength: int = len(ep)
            t1: int = np.random.randint(0, eplength - 1)
            t2: int
            if self._only_trailing_segments:
                t2 = eplength
            else:
                t2 = np.random.randint(t1 + 1, eplength + 1)  # +1 to not include low but include high

            state = uncompressed.states[t1]
            command = Command(reward=sum(uncompressed.rewards[t1:t2]), horizon=t2 - t1)
            action = uncompressed.actions[t1]

            ds.add(state, self._create_command_tensor(command), action)

        return ds

    def _train_behavior_function(self) -> Float:
        ds = self._build_train_dataset(self._batch_size * self._n_updates_per_iter)
        loader = DataLoader(ds, batch_size=self._batch_size, shuffle=True)
        losses: List[float] = []

        with tqdm.tqdm(total=self._n_updates_per_iter) as progress:
            for states, commands, actions in loader:
                actions = actions.to(self._device)

                self._optimizer.zero_grad()  # TODO: test with set_to_none=True
                y_pred = self._behavior(states, commands)  # pylint is drunk  # pylint: disable=not-callable
                pred_loss = -self._behavior.compute_loss(y_pred, actions).mean()
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

    def _evaluate(self, env_step_callback: Optional[EnvStepCallback] = None) -> float:
        return self._generate_episodes(1, True, False, self._evaluation_command, env_step_callback)[0]

    def _generate_episodes(self, num: int, use_bf: bool, add_to_replay_buffer: bool,
                           command: Optional[Command] = None,
                           env_step_callback: Optional[EnvStepCallback] = None) -> List[float]:
        """Plays N episodes using random actions or the behavior function and optionally adds it to the replay buffer."""
        # NOTE: This function got a bit large and complicated, but it's still more clear and
        # readable than any attempts of refactoring it.

        actions_out: Tensor
        commands: List[Command] = []
        traj_states: List[List[Tensor]] = [ [] for _ in range(self._env.num_envs) ]
        traj_actions: List[List[Tensor]] = [ [] for _ in range(self._env.num_envs) ]
        traj_rewards: List[List[float]] = [ [] for _ in range(self._env.num_envs) ]
        summed_rewards: List[float] = []
        progress: Optional[tqdm.tqdm] = tqdm.tqdm(total=num) if num > 1 else None

        # Initialize commands
        if use_bf:
            if command is None:
                command = self._sample_exploratory_command()
            commands = [ command.duplicate() for _ in range(self._env.num_envs) ]

        # Reset
        obss: VecEnvObs = self._env.reset()
        for i, obs in enumerate(obss):
            traj_states[i].append(self._obs_to_tensor(obs))

        while True:
            actions_out = self._explore_act([ s[-1] for s in traj_states ] if use_bf else None,
                                            commands if use_bf else None)
            obss, rewards, dones, _infos = self._env.step(actions_out.numpy())

            if env_step_callback is not None:
                env_step_callback(locals(), globals())

            # Update trajectories and commands, check if done
            for i, (obs, rew, done, action) in enumerate(zip(obss, rewards, dones, actions_out)):
                traj_actions[i].append(action)
                traj_rewards[i].append(rew)

                # If episode is done, add to replay buffer, decrease counter, and return if finished
                if done:
                    t = Trajectory(traj_states[i], traj_actions[i], traj_rewards[i])
                    summed_rewards.append(t.summed_rewards)
                    traj_states[i].clear()
                    traj_actions[i].clear()
                    traj_rewards[i].clear()

                    if add_to_replay_buffer:
                        self._replay_buffer.add_trajectories([ t ])

                    # Return if enough episodes played
                    num -= 1

                    if progress:
                        progress.update()

                    if num <= 0:
                        if progress:
                            progress.close()
                        return summed_rewards

                # Update command
                if use_bf:
                    if done:
                        commands[i] = self._sample_exploratory_command()
                    else:
                        commands[i].update(rew, self._max_reward)

                # Append new state, starting a new state-action-reward tuple
                traj_states[i].append(self._obs_to_tensor(obs))

    def _explore_act(self, states: Optional[List[Tensor]], commands: Optional[List[Command]]) -> Tensor:
        """Returns actions for the given states. If inputs are None, it will sample random actions."""
        if states is not None and commands is not None:
            with torch.no_grad():
                # Combine current states and commands into one stacked tensor each
                actions_out = self._behavior.action(torch.stack(states),
                                                    self._create_vec_command_tensor(commands))
        else:
            samples = np.array([ self._env.action_space.sample() for _ in range(self._env.num_envs) ])
            actions_out = torch.tensor(samples)

        return actions_out.cpu().float()

    def _run_upside_down(self, max_iterations: int, env_step_callback: Optional[EnvStepCallback] = None) -> TrainStats:
        """Algorithm 1 - Upside-Down Reinforcement Learning"""
        stats = self._training_stats

        for train_iter in range(1, max_iterations + 1):
            logging.info("Training...")
            bf_loss = self._train_behavior_function()

            logging.info("Exploring %s episodes...", self._n_episodes_per_iter)
            self._generate_episodes(self._n_episodes_per_iter, True, True, env_step_callback=env_step_callback)

            logging.info("Evaluating...")
            ep_rewards = self._evaluate(env_step_callback)

            stats.loss_history.append(bf_loss)
            stats.command_history.append(self._evaluation_command)
            stats.reward_history.append(ep_rewards)
            average_100_reward = np.mean(stats.reward_history[-100:])
            stats.average_100_reward.append(average_100_reward)

            # pylint: disable=logging-fstring-interpolation
            logging.info(f"""
|--------------------------------------------------
| Iteration: {train_iter}/{max_iterations}
| Evaluated reward: {ep_rewards:.2f}
| Mean 100 evaluated rewards: {average_100_reward:.2f}
| Evaluation command: {self._evaluation_command}
| Training loss: {bf_loss:.2f}
|--------------------------------------------------""")

        return stats

    def _create_command_tensor(self, command: Command) -> Tensor:
        """Creates a command tensor."""
        return torch.tensor((command.reward * self._return_scale,
                             command.horizon * self._horizon_scale)).float()

    def _create_vec_command_tensor(self, commands: List[Command]) -> Tensor:
        return torch.stack([ self._create_command_tensor(c) for c in commands ])

    # Copied and slightly adapted from stable_baselines3/common/policies.py: BaseModel
    # NOTE: This does not handle preprocessing, because we want to retain the original data type as
    # far as possible to reduce memory usage. E.g. a normalized image observation (float) consumes
    # four times more memory than the orignal (uint8).
    # Hence, we do preprocessing in UDRLBehavior.forward()
    def _obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]],
                       add_batch_dimension: bool = False) -> Tensor:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.

        :param observation: the input observation
        :return: The observation as PyTorch tensor
        """
        observation_space: gym.spaces.Space = self._env.observation_space

        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = observation_space.spaces[key]  # pylint: disable=no-member
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                # Add batch dimension if needed
                if add_batch_dimension:
                    observation[key] = obs_.reshape((-1,) + observation_space[key].shape)  # pylint: disable=unsubscriptable-object

        elif is_image_space(observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            # Add batch dimension if needed
            if add_batch_dimension:
                observation = observation.reshape((-1,) + observation_space)

        # NOTE: In our workflow we don't want to move the tensor to the GPU right away when this
        # function gets called.
        obs_tensor = obs_as_tensor(observation, "cpu")  # type: ignore[arg-type]
        # NOTE: We don't support dict observations at the moment
        return obs_tensor  # type: ignore[return-value]
