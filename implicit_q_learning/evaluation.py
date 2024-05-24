from typing import Dict

import flax.linen as nn
import gym
import numpy as np

from tqdm import tqdm


def evaluate(
    agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00, log_video: bool = False
) -> Dict[str, float]:
    # stats = {"return": [], "length": []}
    stats = {}

    sum_reward = 0
    log_phases = []
    log_videos = []
    
    for ep_idx in tqdm(range(num_episodes)):
        observation, done = env.reset(), False
        phase = 0
        ep_video = []
        while not done:
            obs_without_rgb = {k: v for k, v in observation.items() if k != 'color_image1' and k != 'color_image2'}
            action = agent.sample_actions(obs_without_rgb, temperature=temperature)
            observation, rew, done, info = env.step(action)
            phase = max(phase, info['phase'])
            sum_reward += rew
            
            if log_video and ep_idx < 2:
                # Save first two episodes.
                # Make it channel first.
                ep_video.append(observation['color_image2'])

        log_phases.append(phase)
        if ep_video != []:
            # (T, C, H, W)
            ep_video = np.array(ep_video)
            ep_video = np.transpose(ep_video, (0, 3, 1, 2))
            log_videos.append(ep_video)

        # for k in stats.keys():
        #     stats[k].append(info["episode"][k])
        print("Sum of rewards: ", sum_reward)

    # print("Sum of rewards: ", sum_reward)
    # for k, v in stats.items():
        # stats[k] = np.mean(v)
    stats['sum_of_reward'] = sum_reward
    stats['success_rate'] = sum_reward / num_episodes
    stats['phase'] = np.mean(log_phases)

    return stats, log_videos
