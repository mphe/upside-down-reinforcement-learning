#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gymnasium as gym
import logging
from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore  # pyright is buggy
from stable_baselines3.common.env_util import make_vec_env
from UDRL import UDRL, Command


def make_env(*args, **kwargs):
    env = gym.make('ALE/Pong-v5', *args, obs_type="rgb", full_action_space=False, render_mode="rgb_array", **kwargs)
    env = WarpFrame(env)
    env = MaxAndSkipEnv(env)
    return env


def create_udrl(env) -> UDRL:
    return UDRL(
        env,
        "cnn",
        n_warm_up_episodes=50,
        n_updates_per_iter=100,
        n_episodes_per_iter=15,
        batch_size=256,
        last_few=10,
        replay_size=300,
        horizon_scale=2e-2,
        return_scale=1e-2,
        learning_rate=1e-15,
        compress_replay_buffer=False,
        only_trailing_segments=True,
        weighted_replay_sampling=False,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    udrl = create_udrl(env)
    try:
        udrl.learn(400)
    finally:
        udrl.save("pong.pt")
        stats = udrl.get_training_stats()
        stats.to_pandas().to_csv("pong.csv")
        stats.plot()

    del udrl

    udrl = create_udrl(env)
    udrl.load("pong.pt")

    obs = env.reset()

    for _ in range(1000):
        action = udrl.action(obs, [ Command(50, 50) ])
        obs, _rewards, _dones, _info = env.step(action)
        env.render(mode="human")

    env.close()


if __name__ == "__main__":
    main()
