from pathlib import Path

import torch
import numpy as np

from .rollout import RolloutRunner
from ..utils import Logger, Normalizer, Once
from ..utils.pytorch import to_tensor, sync_network


class BaseAgent(torch.nn.Module):
    """Base class for agents."""

    def __init__(self, cfg, ob_space=None):
        super().__init__()
        self._cfg = cfg

        if ob_space is not None:  # TODO: why and who added this line?
            self._ob_norm = Normalizer(
                ob_space,
                eps=1e-3,
                clip_range=cfg.clip_range,
                clip_obs=cfg.clip_obs,
            )
        self._buffer = None
        self._device = torch.device(cfg.device)
        self._step = 0
        self.warm_up_training = Once(False)

    def set_step(self, step):
        self._step = step

    def normalize(self, ob):
        """Normalizes observations."""
        if self._cfg.ob_norm:
            return self._ob_norm.normalize(ob)
        return ob

    def act(self, ob, state=None, is_train=True):
        """Returns action and the actor's activation given an observation `ob`."""
        if hasattr(self, "_rl_agent"):
            return self._rl_agent.act(ob, state, is_train)

        ob = self.normalize(ob)
        ob = ob.copy()
        for k, v in ob.items():
            ob[k] = np.expand_dims(ob[k], axis=0).copy()

        self._actor.eval()
        with torch.no_grad():
            ob = to_tensor(ob, self._device)
            ac = self._actor.act(ob, deterministic=not is_train)[0]
        self._actor.train()

        for k in ac.keys():
            ac[k] = ac[k].cpu().numpy().squeeze(0)
        return ac, None

    def update_normalizer(self, obs=None):
        """Updates normalizers."""
        if self._cfg.ob_norm:
            if obs is None:
                for i in range(len(self._dataset)):
                    self._ob_norm.update(self._dataset[i]["ob"])
                self._ob_norm.recompute_stats()
            else:
                self._ob_norm.update(obs)
                self._ob_norm.recompute_stats()

    def store_episode(self, rollouts):
        """Stores `rollouts` to replay buffer."""
        raise NotImplementedError

    def get_runner(self, cfg, env, env_eval):
        """Returns rollout runner."""
        return RolloutRunner(cfg, env, env_eval, self)

    def is_off_policy(self):
        raise NotImplementedError

    @property
    def buffer(self):
        return self._buffer

    def set_buffer(self, buffer):
        self._buffer = buffer

    def save_replay_buffer(self, replay_dir, ckpt_num):
        """Saves new experience in replay buffer to a file."""
        if not hasattr(self, "_last_ckpt_num"):
            self._last_ckpt_num = -1
        replay_path = (
            Path(replay_dir) / f"replay_{self._last_ckpt_num+1:011d}_{ckpt_num:011d}.pt"
        )
        torch.save(self._buffer.state_dict(), replay_path)
        Logger.warning(f"Save replay buffer: {replay_path}")
        self._last_ckpt_num = ckpt_num

    def load_replay_buffer(self, replay_dir, ckpt_num):
        """Loads replay buffer files up to `ckpt_num`."""
        replay_paths = sorted(Path(replay_dir).glob("replay_*.pt"))
        for replay_path in replay_paths:
            replay_path = str(replay_path)
            if ckpt_num < int(replay_path.rsplit(".")[-2].split("_")[-2]):
                continue
            Logger.warning(f"Load replay_buffer {replay_path}")
            state_dict = torch.load(replay_path)
            self._buffer.append_state_dict(state_dict)

    def set_reward_function(self, predict_reward):
        self._predict_reward = predict_reward

    def sync_networks(self):
        sync_network(self, self._device)

    def update(self):
        raise NotImplementedError
