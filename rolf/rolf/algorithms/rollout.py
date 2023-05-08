"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

from collections import defaultdict

import numpy as np
import cv2

from ..utils import Logger, Info, Every
from ..utils.gym_env import zero_value


class Rollout(object):
    """Rollout storing an episode."""

    def __init__(self, keys=None, precision=32):
        """Initialize buffer."""
        self._history = defaultdict(list)
        self._keys = keys or ["ob", "ob_next", "ac", "rew", "done", "done_mask"]
        self._precision = precision

    def _convert(self, value):
        if isinstance(value, dict):
            return {k: self._convert(v) for k, v in value.items()}

        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32}[self._precision]
        else:
            dtype = value.dtype
        return value.astype(dtype)

    def add(self, data):
        """Add a transition `data` to rollout buffer."""
        for key, value in data.items():
            self._history[key].append(self._convert(value))

    def get(self):
        """Returns rollout buffer and clears buffer."""
        batch = {k: self._history[k] for k in self._keys}
        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
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
        il = hasattr(agent, "predict_reward")

        if env is None:
            while True:
                yield None, 1, {}

        # Initialize rollout buffer.
        rollout = Rollout()
        reward_info = Info()
        rollout_len = 0
        ep_info = Info()
        episode = 0

        while True:
            done = False
            ep_len, ep_rew, ep_rew_rl, ep_rew_il = 0, 0, 0, 0
            ob_next = env.reset()
            state_next = None

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

                if il:
                    reward_il = agent.predict_reward(ob, ob_next, ac)
                    reward_rl = (
                        (1 - cfg.rolf.gail_env_reward) * reward_il
                        + cfg.rolf.gail_env_reward * reward * cfg.rolf.reward_scale
                    )
                else:
                    reward_rl = reward * cfg.rolf.reward_scale

                step += 1
                ep_len += 1
                ep_rew += reward
                ep_rew_rl += reward_rl
                if il:
                    ep_rew_il += reward_il

                # -1 absorbing, 0 done, 1 not done
                done_mask = 0 if done and ep_len < env.max_episode_steps else 1

                rollout.add(dict(ob=ob, ob_next=ob_next, ac=ac, done=done, rew=reward))
                rollout.add(dict(done_mask=done_mask))
                rollout_len += 1

                reward_info.add(info)

                if cfg.env.absorbing_state and done_mask == 0:
                    absorbing_state = env.get_absorbing_state()
                    absorbing_action = zero_value(env.action_space)
                    rollout._history["ob"][-1] = absorbing_state
                    rollout.add(dict(ob=absorbing_state, ac=absorbing_action))
                    rollout.add(dict(done=0, rew=0.0, done_mask=-1))
                    rollout_len += 1

                if every_steps(step):
                    yield rollout.get(), rollout_len, ep_info.get_dict(only_scalar=True)
                    rollout_len = 0

            # Compute average/sum of information.
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            reward_info_dict.update({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})
            if il:
                reward_info_dict["rew_il"] = ep_rew_il
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
        il = hasattr(agent, "predict_reward")

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = Info()

        done = False
        ep_len, ep_rew, ep_rew_rl, ep_rew_il = 0, 0, 0, 0
        ob_next = env.reset()
        state_next = None

        record_frames = []
        if record_video:
            record_frames.append(self._render_frame(ep_len, ep_rew))

        # Rollout one episode.
        while not done:
            ob, state = ob_next, state_next

            # Sample action from policy.
            ac, state_next = agent.act(ob, state, is_train=False)

            # Take a step.
            ob_next, reward, done, info = env.step(ac)
            # print(ac)

            if il:
                reward_il = agent.predict_reward(ob, ob_next, ac)
                reward_rl = (
                    (1 - cfg.rolf.gail_env_reward) * reward_il
                    + cfg.rolf.gail_env_reward * reward * cfg.rolf.reward_scale
                )
            else:
                reward_rl = reward * cfg.rolf.reward_scale

            rollout.add(dict(ob=ob, ob_next=ob_next, ac=ac, done=done, rew=reward))

            ep_len += 1
            ep_rew += reward
            ep_rew_rl += reward_rl
            if il:
                ep_rew_il += reward_il

            reward_info.add(info)
            record_video = False
            if record_video:
                if il:
                    info.update(
                        dict(ep_rew_il=ep_rew_il, rew_il=reward_il, rew_rl=reward_rl)
                    )
                record_frames.append(self._render_frame(ep_len, ep_rew, info))

        # Compute average/sum of information.
        ep_info = {"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl}
        if il:
            ep_info["rew_il"] = ep_rew_il
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

    def _render_frame(self, ep_len, ep_rew, info={}):
        """Renders a video frame and adds caption."""
        frame = self._env_eval.render("rgb_array")
        if len(frame.shape) == 4:
            frame = frame[0]
        if np.max(frame) <= 1.0:
            frame *= 255.0
        frame = frame.astype(np.uint8)

        # Add caption to video frame.
        if self._cfg.record_video_caption:
            # Set the minimum size of frames to (512, 512) for caption readibility.
            if frame.shape[0] < 512:
                frame = cv2.resize(frame, (512, 512))
            h, w = frame.shape[:2]

            # Add caption.
            frame = np.concatenate([frame, np.zeros((h, w, 3), np.uint8)], 0)
            scale = h / 512
            font_size = 0.4 * scale
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            x, y = int(5 * scale), h + int(10 * scale)
            add_text = lambda x, y, c, t: cv2.putText(
                frame, t, (x, y), font_face, font_size, c, 1, cv2.LINE_AA
            )

            add_text(x, y, (255, 255, 0), f"{ep_len:5} {ep_rew}")
            for i, k in enumerate(info.keys()):
                key_text = f"{k}: "
                key_width = cv2.getTextSize(key_text, font_face, font_size, 1)[0][0]
                offset = int(12 * scale) * (i + 2)
                add_text(x, y + offset, (66, 133, 244), key_text)
                add_text(x + key_width, y + offset, (255, 255, 255), str(info[k]))

        return frame
