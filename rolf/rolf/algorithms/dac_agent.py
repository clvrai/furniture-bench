from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.distributions
from torch.optim.lr_scheduler import StepLR
import gym.spaces

from .base_agent import BaseAgent
from .ddpg_agent import DDPGAgent
from .sac_agent import SACAgent
from .dataset import ReplayBuffer, RandomSampler
from .expert_dataset import ExpertDataset
from ..networks.discriminator import Discriminator
from ..utils import Logger, Info
from ..utils.mpi import mpi_average
from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    sync_network,
    sync_grad,
    to_tensor,
    dictlist_to_tensor,
)


class DACAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        cfg.rl_algo.device = cfg.device
        if cfg.rl_algo.name == "td3":
            self._rl_agent = DDPGAgent(cfg.rl_algo, ob_space, ac_space)
        elif cfg.rl_algo.name == "sac":
            self._rl_agent = SACAgent(cfg.rl_algo, ob_space, ac_space)
        else:
            raise ValueError(f"rl_algo={cfg.rl_algo.name} is not supported for DAC.")
        self._rl_agent.set_reward_function(self._predict_reward)

        # build up networks
        self._discriminator = Discriminator(
            ob_space, ac_space if not cfg.gail_no_action else None
        )
        self._discriminator_loss = nn.BCEWithLogitsLoss()
        self.to(self._device)

        # build optimizers
        self._discriminator_optim = optim.Adam(
            self._discriminator.parameters(), lr=cfg.discriminator_lr
        )

        # build learning rate scheduler
        self._discriminator_lr_scheduler = StepLR(
            self._discriminator_optim, step_size=cfg.max_global_step // 5, gamma=0.5
        )

        # expert dataset
        if cfg.is_train:
            self._dataset = ExpertDataset(
                cfg.demo_path,
                cfg.demo_subsample_interval,
                ac_space,
                use_low_level=cfg.demo_low_level,
                sample_range=cfg.demo_sample_range,
            )
            if cfg.absorbing_state:
                self._dataset.add_absorbing_states(ob_space, ac_space)
            self._data_loader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=True,
            )
            self._data_iter = iter(self._data_loader)

        # per-episode replay buffer
        sampler = RandomSampler()
        buffer_keys = ["ob", "ob_next", "ac", "done", "done_mask", "rew"]
        self._buffer = ReplayBuffer(buffer_keys, cfg.buffer_size, sampler.sample_func)

        self._rl_agent.set_buffer(self._buffer)

        self._update_iter = 0

        self._log_creation()

    def _predict_reward(self, ob, ob_next, ac):
        if not self._cfg.gail_use_action:
            ac = None
        if not self._cfg.gail_use_next_ob:
            ob_next = None

        self._discriminator.eval()
        with torch.no_grad():
            ret = self._discriminator(ob, ob_next, ac)
            eps = 1e-10
            s = torch.sigmoid(ret)
            if self._cfg.gail_reward == "vanilla":
                reward = -(1 - s + eps).log()
            elif self._cfg.gail_reward == "gan":
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self._cfg.gail_reward == "d":
                reward = ret
            elif self._cfg.gail_reward == "amp":
                ret = torch.clamp(ret, 0, 1) - 1
                reward = 1 - ret**2
        self._discriminator.train()
        return reward

    def predict_reward(self, ob, ob_next=None, ac=None):
        def get_tensor(v):
            if isinstance(v, list):
                return dictlist_to_tensor(v, self._device)
            else:
                return to_tensor(v, self._device)

        ob = get_tensor(self.normalize(ob))
        if self._cfg.gail_use_next_ob:
            ob_next = get_tensor(self.normalize(ob_next))
        else:
            ob_next = None

        if self._cfg.gail_use_action:
            ac = get_tensor(ac)
        else:
            ac = None

        reward = self._predict_reward(ob, ob_next, ac)
        return reward.cpu().item()

    def _log_creation(self):
        Logger.info("Creating a DAC agent")
        Logger.info(
            f"The discriminator has {count_parameters(self._discriminator)} parameters"
        )

    def is_off_policy(self):
        return True

    def store_episode(self, rollouts):
        self._rl_agent.store_episode(rollouts)

    def state_dict(self):
        return {
            "rl_agent": self._rl_agent.state_dict(),
            "discriminator_state_dict": self._discriminator.state_dict(),
            "discriminator_optim_state_dict": self._discriminator_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "rl_agent" in ckpt:
            self._rl_agent.load_state_dict(ckpt["rl_agent"])
        else:
            self._rl_agent.load_state_dict(ckpt)
            self.to(self._device)
            return

        self._discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self.to(self._device)

        self._discriminator_optim.load_state_dict(
            ckpt["discriminator_optim_state_dict"]
        )
        optimizer_cuda(self._discriminator_optim, self._device)

    def update(self):
        train_info = Info()

        if self._update_iter % self._cfg.discriminator_update_freq == 0:
            self._num_updates = 1
            for _ in range(self._num_updates):
                policy_data = self._buffer.sample(self._cfg.batch_size)
                try:
                    expert_data = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    expert_data = next(self._data_iter)
                _train_info = self._update_discriminator(policy_data, expert_data)
                train_info.add(_train_info)
            self._discriminator_lr_scheduler.step()

        _train_info = self._rl_agent.train()
        train_info.add(_train_info)

        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, policy_data, expert_data):
        info = Info()

        _to_tensor = lambda x: to_tensor(x, self._device)
        # pre-process observations
        p_o = policy_data["ob"]
        p_o = self.normalize(p_o)

        p_bs = len(policy_data["ac"])
        p_o = _to_tensor(p_o)
        if self._cfg.gail_no_action:
            p_ac = None
        else:
            p_ac = _to_tensor(policy_data["ac"])

        e_o = expert_data["ob"]
        e_o = self.normalize(e_o)

        e_bs = len(expert_data["ac"])
        e_o = _to_tensor(e_o)
        if self._cfg.gail_no_action:
            e_ac = None
        else:
            e_ac = _to_tensor(expert_data["ac"])

        p_logit = self._discriminator(p_o, p_ac)
        e_logit = self._discriminator(e_o, e_ac)

        p_output = torch.sigmoid(p_logit)
        e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.zeros_like(p_logit).to(self._device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.ones_like(e_logit).to(self._device)
        )

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits).entropy().mean()
        entropy_loss = -self._cfg.gail_entropy_loss_coeff * entropy

        grad_pen = self._compute_grad_pen(p_o, p_ac, e_o, e_ac)
        grad_pen_loss = self._cfg.gail_grad_penalty_coeff * grad_pen

        gail_loss = p_loss + e_loss + entropy_loss + grad_pen_loss

        # update the discriminator
        self._discriminator.zero_grad()
        gail_loss.backward()
        sync_grad(self._discriminator, self._device)
        self._discriminator_optim.step()

        info["gail_policy_output"] = p_output.mean().detach().cpu().item()
        info["gail_expert_output"] = e_output.mean().detach().cpu().item()
        info["gail_entropy"] = entropy.detach().cpu().item()
        info["gail_policy_loss"] = p_loss.detach().cpu().item()
        info["gail_expert_loss"] = e_loss.detach().cpu().item()
        info["gail_entropy_loss"] = entropy_loss.detach().cpu().item()
        info["gail_grad_pen"] = grad_pen.detach().cpu().item()
        info["gail_grad_loss"] = grad_pen_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))

    def _compute_grad_pen(self, policy_ob, policy_ac, expert_ob, expert_ac):
        batch_size = self._cfg.batch_size
        alpha = torch.rand(batch_size, 1, device=self._device)

        def blend_dict(a, b, alpha):
            if isinstance(a, dict):
                return OrderedDict(
                    [(k, blend_dict(a[k], b[k], alpha)) for k in a.keys()]
                )
            elif isinstance(a, list):
                return [blend_dict(a[i], b[i], alpha) for i in range(len(a))]
            else:
                expanded_alpha = alpha.expand_as(a)
                ret = expanded_alpha * a + (1 - expanded_alpha) * b
                ret.requires_grad = True
                return ret

        interpolated_ob = blend_dict(policy_ob, expert_ob, alpha)
        inputs = list(interpolated_ob.values())
        if policy_ac is not None:
            interpolated_ac = blend_dict(policy_ac, expert_ac, alpha)
            inputs = inputs + list(interpolated_ob.values())
        else:
            interpolated_ac = None

        interpolated_logit = self._discriminator(interpolated_ob, interpolated_ac)
        ones = torch.ones(interpolated_logit.size(), device=self._device)

        grad = autograd.grad(
            outputs=interpolated_logit,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
