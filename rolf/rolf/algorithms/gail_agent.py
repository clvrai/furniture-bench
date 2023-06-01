from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.distributions
from torch.optim.lr_scheduler import StepLR

from .base_agent import BaseAgent
from .ppo_agent import PPOAgent
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


class GAILAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        cfg.rl_algo.device = cfg.device
        if cfg.rl_algo.name == "ppo":
            self._rl_agent = PPOAgent(cfg.rl_algo, ob_space, ac_space)
        else:
            raise ValueError(f"rl_algo={cfg.rl_algo.name} is not supported for GAIL.")
        self._rl_agent.set_reward_function(self.predict_reward)

        # build up networks
        self._discriminator = Discriminator(
            ob_space,
            ob_space if cfg.gail_use_next_ob else None,
            ac_space if cfg.gail_use_action else None,
        )
        if cfg.discriminator_loss_type == "gan":
            self._discriminator_loss = nn.BCEWithLogitsLoss()
        elif cfg.discriminator_loss_type == "lsgan":
            self._discriminator_loss = nn.MSELoss()
        self.to(self._device)

        # build optimizers
        self._discriminator_optim = optim.Adam(
            self._discriminator.parameters(), lr=cfg.discriminator_lr
        )

        # build learning rate scheduler
        self._discriminator_lr_scheduler = StepLR(
            self._discriminator_optim,
            step_size=cfg.max_global_step // cfg.rl_algo.train_every,
            gamma=0.5,
        )

        # expert dataset
        self._dataset = ExpertDataset(
            cfg.demo_path,
            cfg.demo_subsample_interval,
            ac_space,
            use_low_level=cfg.demo_low_level,
            sample_range=cfg.demo_sample_range,
        )
        self._data_loader = torch.utils.data.DataLoader(
            self._dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        self._data_iter = iter(self._data_loader)

        # policy dataset
        sampler = RandomSampler()
        keys = ["ob"]
        if cfg.gail_use_action:
            keys.append("ac")
        if cfg.gail_use_next_ob:
            keys.append("ob_next")

        self._buffer = ReplayBuffer(
            keys, cfg.discriminator_buffer_size, sampler.sample_func
        )

        # update observation normalizer with dataset
        self.update_normalizer()

        self._log_creation()

    def set_buffer(self, buffer):
        self._buffer = buffer
        self._rl_agent.set_buffer(buffer)

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
        return reward.cpu().detach().numpy().squeeze()

    def _log_creation(self):
        Logger.info("Creating a GAIL agent")
        Logger.info(
            f"The discriminator has {count_parameters(self._discriminator)} parameters"
        )

    def is_off_policy(self):
        return False

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)
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

    def update_normalizer(self, obs=None):
        """Updates normalizers for discriminator and PPO agent."""
        if self._cfg.ob_norm:
            if obs is None:
                if self._cfg.is_train:
                    data_loader = torch.utils.data.DataLoader(
                        self._dataset,
                        batch_size=self._cfg.batch_size,
                        shuffle=False,
                        drop_last=False,
                    )
                    for obs in data_loader:
                        super().update_normalizer(obs)
                        self._rl_agent.update_normalizer(obs)
            else:
                super().update_normalizer(obs)
                self._rl_agent.update_normalizer(obs)
                try:
                    expert_data = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    expert_data = next(self._data_iter)
                super().update_normalizer(expert_data["ob"])
                self._rl_agent.update_normalizer(expert_data["ob"])

    def update(self):
        train_info = Info()

        num_batches = (
            self._cfg.train_every
            // self._cfg.batch_size
            // self._cfg.discriminator_update_freq
        )
        assert num_batches > 0
        for _ in range(num_batches):
            policy_data = self._buffer.sample(self._cfg.batch_size)
            try:
                expert_data = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self._data_loader)
                expert_data = next(self._data_iter)

            _train_info = self._update_discriminator(policy_data, expert_data)
            train_info.add(_train_info)

        _train_info = self._rl_agent.update()
        train_info.add(_train_info)

        self._discriminator_lr_scheduler.step()

        if not self._cfg.discriminator_replay_buffer:
            self._buffer.clear()
        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, policy_data, expert_data):
        info = Info()
        _to_tensor = lambda x: to_tensor(x, self._device)

        p_o = self.normalize(policy_data["ob"])
        p_o = _to_tensor(p_o)

        e_o = self.normalize(expert_data["ob"])
        e_o = _to_tensor(e_o)

        if self._cfg.gail_use_next_ob:
            p_o_next = self.normalize(policy_data["ob_next"])
            p_o_next = _to_tensor(p_o_next)

            e_o_next = self.normalize(expert_data["ob_next"])
            e_o_next = _to_tensor(e_o_next)
        else:
            p_o_next = None
            e_o_next = None

        if self._cfg.gail_use_action:
            p_ac = _to_tensor(policy_data["ac"])
            e_ac = _to_tensor(expert_data["ac"])
        else:
            p_ac = None
            e_ac = None

        p_logit = self._discriminator(p_o, p_o_next, p_ac)
        e_logit = self._discriminator(e_o, e_o_next, e_ac)

        if self._cfg.discriminator_loss_type == "lsgan":
            p_output = p_logit
            e_output = e_logit
        else:
            p_output = torch.sigmoid(p_logit)
            e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.zeros_like(p_logit).to(self._device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.ones_like(e_logit).to(self._device)
        )

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
        entropy_loss = -self._cfg.gail_entropy_loss_coeff * entropy

        grad_pen = self._compute_grad_pen(p_o, p_o_next, p_ac, e_o, e_o_next, e_ac)
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
        info["gail_loss"] = gail_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))

    def _compute_grad_pen(
        self, policy_ob, policy_ob_next, policy_ac, expert_ob, expert_ob_next, expert_ac
    ):
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

        if policy_ob_next is not None:
            interpolated_ob_next = blend_dict(policy_ob_next, expert_ob_next, alpha)
            inputs = inputs + list(interpolated_ob_next.values())
        else:
            interpolated_ob_next = None

        if policy_ac is not None:
            interpolated_ac = blend_dict(policy_ac, expert_ac, alpha)
            inputs = inputs + list(interpolated_ac.values())
        else:
            interpolated_ac = None

        interpolated_logit = self._discriminator(
            interpolated_ob, interpolated_ob_next, interpolated_ac
        )
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
