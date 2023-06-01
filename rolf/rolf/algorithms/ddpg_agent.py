import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym.spaces
from torch.optim.lr_scheduler import StepLR

from .base_agent import BaseAgent
from .dataset import ReplayBuffer, RandomSampler
from ..networks import Actor, Critic
from ..utils import Logger, Info
from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    sync_grad,
    to_tensor,
    scale_dict_tensor,
    copy_network,
    soft_copy_network,
)


class DDPGAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        # build up networks
        self._actor = Actor(cfg, ob_space, ac_space, cfg.tanh_policy)
        self._critic = Critic(cfg, ob_space, ac_space)

        # build up target networks
        self._actor_target = Actor(cfg, ob_space, ac_space, cfg.tanh_policy)
        self._critic_target = Critic(cfg, ob_space, ac_space)
        self.to(self._device)
        copy_network(self._actor_target, self._actor)
        copy_network(self._critic_target, self._critic)
        self._actor.encoder.copy_conv_weights_from(self._critic.encoder)
        self._actor_target.encoder.copy_conv_weights_from(self._critic_target.encoder)

        # build optimizers
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=cfg.actor_lr)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=cfg.critic_lr)

        # build learning rate scheduler
        self._actor_lr_scheduler = StepLR(
            self._actor_optim,
            step_size=self._cfg.max_global_step // 5,
            gamma=0.5,
        )

        # per-episode replay buffer
        sampler = RandomSampler()
        buffer_keys = ["ob", "ob_next", "ac", "done", "done_mask", "rew"]
        self._buffer = ReplayBuffer(buffer_keys, cfg.buffer_size, sampler.sample_func)

        self._update_iter = 0
        self._predict_reward = None

        self._log_creation()

    def _log_creation(self):
        Logger.info("Creating a DDPG agent")
        Logger.info("The actor has %d parameters", count_parameters(self._actor))
        Logger.info("The critic has %d parameters", count_parameters(self._critic))

    def is_off_policy(self):
        return True

    def act(self, ob, state=None, is_train=True):
        """Returns action and the actor's activation given an observation @ob."""
        ac = super().act(ob, state, is_train=is_train)[0]

        if not is_train:
            return ac, None

        if self._cfg.epsilon_greedy and np.random.rand() < self._cfg.epsilon_greedy_eps:
            for k, v in self._ac_space.spaces.items():
                ac[k] = v.sample()
            return ac, None

        for k, v in self._ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box):
                ac[k] = ac[k] + self._cfg.expl_noise * np.random.randn(*ac[k].shape)
                ac[k] = np.clip(ac[k], v.low, v.high)
                # np.float64 action raises out-of-range exception in gym
                ac[k] = ac[k].astype(np.float32)
        return ac, None

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            "update_iter": self._update_iter,
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "critic_state_dict" not in ckpt:
            missing = self._actor.load_state_dict(
                ckpt["actor_state_dict"], strict=False
            )
            copy_network(self._actor_target, self._actor)
            self.to(self._device)
            return

        self._update_iter = ckpt["update_iter"]
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        copy_network(self._actor_target, self._actor)
        copy_network(self._critic_target, self._critic)
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self.to(self._device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._device)
        optimizer_cuda(self._critic_optim, self._device)

    def update(self):
        train_info = Info()

        self._num_updates = 1
        for _ in range(self._num_updates):
            transitions = self._buffer.sample(self._cfg.batch_size)
            train_info.add(self._update_network(transitions))

        return train_info.get_dict()

    def _update_actor(self, o, mask):
        info = Info()

        # the actor loss
        actions_real = self._actor.act(o, detach_conv=True)[0]

        q_pred = self._critic(o, actions_real, detach_conv=True)
        if self._cfg.critic_ensemble > 1:
            q_pred = q_pred[0]

        if self._cfg.absorbing_state:
            # do not update the actor for absorbing states
            a_mask = 1.0 - torch.clamp(-mask, min=0)  # 0 absorbing, 1 done/not done
            actor_loss = -(q_pred * a_mask).sum()
            if a_mask.sum() > 1e-8:
                actor_loss /= a_mask.sum()
        else:
            actor_loss = -q_pred.mean()
        info["actor_loss"] = actor_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), self._cfg.max_grad_norm
            )
        sync_grad(self._actor, self._device)
        self._actor_optim.step()

        return info

    def _update_critic(self, o, ac, rew, o_next, mask):
        info = Info()

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self._actor_target.act(o_next)[0]

            # TD3 adds noise to action
            if self._cfg.critic_ensemble > 1:
                for k in self._ac_space.spaces.keys():
                    noise = (
                        torch.randn_like(actions_next[k]) * self._cfg.policy_noise
                    ).clamp(-self._cfg.policy_noise_clip, self._cfg.policy_noise_clip)
                    actions_next[k] = (actions_next[k] + noise).clamp(-1, 1)

                if self._cfg.absorbing_state:
                    a_mask = torch.clamp(mask, min=0)  # 0 absorbing/done, 1 not done
                    masked_actions_next = scale_dict_tensor(actions_next, a_mask)
                    q_next_values = self._critic_target(o_next, masked_actions_next)
                else:
                    q_next_values = self._critic_target(o_next, actions_next)

                q_next_value = torch.min(*q_next_values)

            else:
                q_next_value = self._critic_target(o_next, actions_next)

            # For IL, use IL reward
            if self._predict_reward is not None:
                rew_il = self._predict_reward(o, o_next, ac)
                rew = (
                    1 - self._cfg.gail_env_reward
                ) * rew_il + self._cfg.gail_env_reward * rew

            if self._cfg.absorbing_state:
                target_q_value = rew + self._cfg.rl_discount * q_next_value
            else:
                target_q_value = rew + mask * self._cfg.rl_discount * q_next_value

        # the q loss
        if self._cfg.critic_ensemble == 1:
            real_q_value = self._critic(o, ac)
            critic_loss = F.mse_loss(target_q_value, real_q_value)
        else:
            real_q_value1, real_q_value2 = self._critic(o, ac)
            critic1_loss = F.mse_loss(target_q_value, real_q_value1)
            critic2_loss = F.mse_loss(target_q_value, real_q_value2)
            critic_loss = critic1_loss + critic2_loss

        # update the critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        sync_grad(self._critic, self._device)
        self._critic_optim.step()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()

        if self._cfg.critic_ensemble == 1:
            info["min_real1_q"] = real_q_value.min().cpu().item()
            info["real1_q"] = real_q_value.mean().cpu().item()
            info["critic1_loss"] = critic_loss.cpu().item()
        else:
            info["min_real1_q"] = real_q_value1.min().cpu().item()
            info["min_real2_q"] = real_q_value2.min().cpu().item()
            info["real1_q"] = real_q_value1.mean().cpu().item()
            info["real2_q"] = real_q_value2.mean().cpu().item()
            info["critic1_loss"] = critic1_loss.cpu().item()
            info["critic2_loss"] = critic2_loss.cpu().item()

        return info

    def _update_network(self, transitions):
        info = Info()

        # pre-process the observation
        o, o_next = transitions["ob"], transitions["ob_next"]
        o = self.normalize(o)
        o_next = self.normalize(o_next)
        bs = len(transitions["done"])

        _to_tensor = lambda x: to_tensor(x, self._device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])
        mask = _to_tensor(transitions["done_mask"]).reshape(bs, 1)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        self._update_iter += 1

        critic_train_info = self._update_critic(o, ac, rew, o_next, mask)
        info.add(critic_train_info)

        if (
            self._update_iter % self._cfg.actor_update_freq == 0
            and self._update_iter > self._cfg.actor_update_delay
        ):
            actor_train_info = self._update_actor(o, mask)
            info.add(actor_train_info)
            self._actor_lr_scheduler.step()

        if self._update_iter % self._cfg.critic_target_update_freq == 0:
            for i, fc in enumerate(self._critic.fcs):
                soft_copy_network(
                    self._critic_target.fcs[i],
                    fc,
                    self._cfg.critic_soft_update_weight,
                )
            soft_copy_network(
                self._critic_target.encoder,
                self._critic.encoder,
                self._cfg.encoder_soft_update_weight,
            )

        if (
            self._update_iter % self._cfg.actor_target_update_freq == 0
            and self._update_iter > self._cfg.actor_update_delay
        ):
            soft_copy_network(
                self._actor_target.fc,
                self._actor.fc,
                self._cfg.actor_soft_update_weight,
            )
            for k, fc in self._actor.fcs.items():
                soft_copy_network(
                    self._actor_target.fcs[k],
                    fc,
                    self._cfg.actor_soft_update_weight,
                )
            soft_copy_network(
                self._actor_target.encoder,
                self._actor.encoder,
                self._cfg.encoder_soft_update_weight,
            )

        return info.get_dict(only_scalar=True)
