from typing import Dict

import flax.linen as nn
import gym
import numpy as np

from tqdm import tqdm


def evaluate(
    agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00
) -> Dict[str, float]:
    # stats = {"return": [], "length": []}
    stats = {}

    sum_reward = 0
    for _ in tqdm(range(num_episodes)):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=temperature)
            observation, rew, done, info = env.step(action)
            sum_reward += rew

        # for k in stats.keys():
        #     stats[k].append(info["episode"][k])
        print("Sum of rewards: ", sum_reward)

    # print("Sum of rewards: ", sum_reward)
    # for k, v in stats.items():
        # stats[k] = np.mean(v)
    stats['sum_of_reward'] = sum_reward
    stats['success_rate'] = sum_reward / num_episodes

    return stats
