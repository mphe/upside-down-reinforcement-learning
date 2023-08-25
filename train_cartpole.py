#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import logging
from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore  # pyright is buggy
from stable_baselines3.common.env_util import make_vec_env
from UDRL import UDRL, Command


def create_udrl(env) -> UDRL:
    return UDRL(
        env,
        "mlp",
        n_warm_up_episodes=50,
        n_updates_per_iter=100,
        n_episodes_per_iter=15,
        batch_size=256,
        last_few=50,
        replay_size=700,
        horizon_scale=2e-2,
        return_scale=2e-2,
        learning_rate=1e-3,
        compress_replay_buffer=False,
        only_trailing_segments=True,
        weighted_replay_sampling=False,
        policy_kwargs=dict(net_arch=[64, 64, 64, 64])
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    env = make_vec_env(lambda: gym.make("CartPole-v1", render_mode="rgb_array"), n_envs=4, vec_env_cls=SubprocVecEnv)

    udrl = create_udrl(env)
    try:
        udrl.learn(500)
    finally:
        udrl.save("cartpole.pt")
        stats = udrl.get_training_stats()
        stats.to_pandas().to_csv("cartpole.csv")
        stats.plot()

    del udrl

    udrl = create_udrl(env)
    udrl.load("cartpole.pt")

    obs = env.reset()

    for _ in range(1000):
        action = udrl.action(obs, [ Command(500, 500) ])
        obs, _rewards, _dones, _info = env.step(action)
        env.render(mode="human")

    env.close()


if __name__ == "__main__":
    main()
