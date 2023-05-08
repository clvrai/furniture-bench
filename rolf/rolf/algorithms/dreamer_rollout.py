"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

import numpy as np
import gym.spaces

from .rollout import Rollout, RolloutRunner
from ..utils import Logger, Info, Every
from ..utils.pytorch import check_memory_kill_switch


class DreamerRolloutRunner(RolloutRunner):
    """Rollout a policy."""

    def __init__(self, cfg, env, env_eval, agent):
        """
        Args:
            cfg: configurations for the environment.
            env: training environment.
            env_eval: testing environment.
            agent: policy.
        """
        self._cfg = cfg
        self._env = env
        self._env_eval = env_eval
        self._agent = agent
        self._exclude_rollout_log = ["episode_success_state"]

    def run(self, every_steps=None, every_episodes=None, log_prefix="", step=0):
        """
        Collects trajectories for training and yield every `every_steps`/`every_episodes`.

        Args:
            every_steps: if not None, returns rollouts `every_steps`
            every_episodes: if not None, returns rollouts `every_epiosdes`
            log_prefix: log as `log_prefix` rollout: %s
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")
        every_steps = Every(every_steps, step)
        every_episodes = Every(every_episodes)

        cfg = self._cfg
        env = self._env
        agent = self._agent

        # Initialize rollout buffer.
        rollout = Rollout(["ob", "ac", "rew", "done"], cfg.precision)
        reward_info = Info()
        ep_info = Info()
        rollout_len = 0
        episode = 0
        dummy_ac = np.zeros(gym.spaces.flatdim(env.action_space))

        while True:
            done = False
            ep_len, ep_rew, ep_rew_rl = 0, 0, 0
            ob_next = env.reset()
            state_next = None

            # Add dummy previous action for the first transition.
            rollout.add(dict(ob=ob_next, ac=dummy_ac, rew=0.0, done=False))

            # Rollout one episode.
            while not done:
                ob, state = ob_next, state_next

                # Sample action from policy.
                if step < cfg.rolf.warm_up_steps:
                    ac, state_next = env.action_space.sample(), None
                else:
                    ac, state_next = agent.act(ob, state, is_train=True)

                # Take a step.
                ob_next, reward, done, info = env.step(ac)

                reward_rl = reward * cfg.rolf.reward_scale

                step += 1
                ep_len += 1
                ep_rew += reward
                ep_rew_rl += reward_rl

                flat_ac = gym.spaces.flatten(env.action_space, ac)
                rollout.add(dict(ob=ob_next, ac=flat_ac, done=done, rew=float(reward)))
                rollout_len += 1
                reward_info.add(info)

                if every_steps(step):
                    yield rollout.get(), rollout_len, ep_info.get_dict(only_scalar=True)
                    rollout_len = 0

            # compute average/sum of information
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            reward_info_dict.update({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})
            ep_info.add(reward_info_dict)

            Logger.info(
                log_prefix + " rollout: %s",
                {
                    k: v
                    for k, v in reward_info_dict.items()
                    if k not in self._exclude_rollout_log and np.isscalar(v)
                },
            )

            episode += 1
            if every_episodes(episode):
                yield rollout.get(), rollout_len, ep_info.get_dict(only_scalar=True)
                rollout_len = 0

    def run_episode(self, record_video=False):
        """
        Runs one episode and returns the rollout for evaluation.

        Args:
            record_video: record video of rollout if True.
        """
        cfg = self._cfg
        env = self._env_eval
        agent = self._agent

        # initialize rollout buffer
        rollout = Rollout(["ob", "ac", "rew", "done"], cfg.precision)
        reward_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ep_rew_rl = 0

        ob_next = env.reset()
        state_next = None

        record_frames = []
        if record_video:
            record_frames.append(self._render_frame(ep_len, ep_rew))

        # to ensure that it doesn't run forever
        max_rollout_len = cfg.env.max_rollout_len

        # run rollout
        while not done and ep_len < max_rollout_len:
            ob = ob_next
            state = state_next

            # sample action from policy
            ac, state_next = agent.act(ob, state, is_train=False)

            # take a step
            ob_next, reward, done, info = env.step(ac)

            reward_rl = reward * cfg.rolf.reward_scale

            flat_ac = gym.spaces.flatten(env.action_space, ac)
            rollout.add(dict(ob=ob, ac=flat_ac, done=done, rew=reward))

            ep_len += 1
            ep_rew += reward
            ep_rew_rl += reward_rl

            reward_info.add(info)
            if record_video:
                frame_info = info.copy()
                record_frames.append(self._render_frame(ep_len, ep_rew, frame_info))

            check_memory_kill_switch()

        # add last observation
        rollout.add({"ob": ob_next})

        # compute average/sum of information
        ep_info = {"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl}
        if "episode_success_state" in reward_info.keys():
            ep_info["episode_success_state"] = reward_info["episode_success_state"]
        ep_info.update(reward_info.get_dict(reduction="sum", only_scalar=True))

        Logger.info(
            "rollout: %s",
            {
                k: v
                for k, v in ep_info.items()
                if k not in self._exclude_rollout_log and np.isscalar(v)
            },
        )
        return rollout.get(), ep_info, record_frames
