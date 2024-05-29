from typing import Dict

import flax.linen as nn
import gym
import numpy as np

from tqdm import tqdm

from dataset_utils import min_max_normalize, max_normalize, replay_chunk_to_seq


def evaluate(
    agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00, log_video: bool = False, reward_model = None, normalization="max", max_rew=None, window_size=None
) -> Dict[str, float]:
    # Hacky import to avoid circular imports.
    # stats = {"return": [], "length": []}
    stats = {}

    sum_reward = 0
    log_phases = []
    log_videos = []
    
    sum_success = 0
    learned_sum_rewards = []
    for ep_idx in tqdm(range(num_episodes)):
        observations = []
        actions = []
        rewards = []

        observation, done = env.reset(), False
        phase = 0
        ep_video = []
        while not done:
            if reward_model is not None:
                observations.append(observation)
            obs_without_rgb = {k: v for k, v in observation.items() if k != 'color_image1' and k != 'color_image2'}
            action = agent.sample_actions(obs_without_rgb, temperature=temperature)
            observation, rew, done, info = env.step(action)
            if reward_model is not None:
                actions.append(action)
                rewards.append(rew)
            phase = max(phase, info['phase'])
            sum_reward += rew
            
            if log_video and ep_idx < 2:
                # Save first two episodes.
                # Make it channel first.
                ep_video.append(observation['color_image2'])

        if reward_model is not None:
            # compute reds reward
            x = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
            }
            seq = reward_model(replay_chunk_to_seq(x, window_size))
            rewards = np.asarray([elem[reward_model.PUBLIC_LIKELIHOOD_KEY] for elem in seq])
            if normalization == "min_max":
                min_max_normalize(rewards)
            if normalization == "max":
                rewards = max_normalize(rewards, max_rew)
            learned_sum_rewards.append(sum(rewards))

        if env.furnitures[0].all_assembled(): # Only single environment.
            sum_success += 1

        log_phases.append(phase)
        if ep_video != []:
            # (T, C, H, W)
            ep_video = np.array(ep_video)
            ep_video = np.transpose(ep_video, (0, 3, 1, 2))
            log_videos.append(ep_video)

        # for k in stats.keys():
        #     stats[k].append(info["episode"][k])
        print("GT sum of rewards: ", sum_reward)

    # print("Sum of rewards: ", sum_reward)
    # for k, v in stats.items():
        # stats[k] = np.mean(v)
    stats['gt_sum_of_reward'] = sum_reward
    # stats['success_rate'] = sum_reward / num_episodes
    stats['success_rate'] = sum_success / num_episodes
    stats['phase'] = np.mean(log_phases)
    if reward_model is not None:
        stats['learned_sum_of_reward'] = np.mean(learned_sum_rewards)

    return stats, log_videos
