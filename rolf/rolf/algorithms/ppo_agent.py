import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ..networks import Actor, Critic
from ..utils import Logger, Info
from ..utils.mpi import mpi_average
from ..utils.pytorch import count_parameters, dictlist_to_tensor, optimizer_cuda
from ..utils.pytorch import sync_grad, to_tensor, copy_network
from .base_agent import BaseAgent
from .dataset import RandomSampler, ReplayBuffer


class PPOAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)

        self._ac_space = ac_space

        # build up networks
        self._actor = Actor(cfg, ob_space, ac_space, cfg.tanh_policy)
        self._old_actor = Actor(cfg, ob_space, ac_space, cfg.tanh_policy)
        self._critic = Critic(cfg, ob_space)
        self.to(self._device)

        self._actor_optim = optim.Adam(
            self._actor.parameters(),
            lr=cfg.actor_lr,
            weight_decay=cfg.actor_weight_decay,
        )
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=cfg.critic_lr)

        self._actor_lr_scheduler = StepLR(
            self._actor_optim,
            step_size=cfg.max_global_step // cfg.train_every,
            gamma=0.5,
        )
        self._critic_lr_scheduler = StepLR(
            self._critic_optim,
            step_size=cfg.max_global_step // cfg.train_every,
            gamma=0.5,
        )

        sampler = RandomSampler()
        self._rollouts = None
        self._buffer = ReplayBuffer(
            ["ob", "ob_next", "ac", "done", "rew", "ret", "adv"],
            cfg.train_every,
            sampler.sample_func,
        )

        self._update_iter = 0

        self._log_creation()

    def _log_creation(self):
        Logger.info("Creating a PPO agent")
        Logger.info(f"The actor has {count_parameters(self._actor)} parameters")
        Logger.info(f"The critic has {count_parameters(self._critic)} parameters")

    def store_episode(self, rollouts):
        self._rollouts = rollouts
        self._compute_gae(rollouts)
        self._buffer.store_episode(rollouts)

    def _compute_gae(self, rollouts):
        T = len(rollouts["done"])
        ob = rollouts["ob"]
        ob = self.normalize(ob)
        ob = dictlist_to_tensor(ob, self._device)

        ob_last = rollouts["ob_next"][-1:]
        ob_last = self.normalize(ob_last)
        ob_last = dictlist_to_tensor(ob_last, self._device)

        done = rollouts["done"]
        rew = rollouts["rew"]

        vpred = self._critic(ob).detach().cpu().numpy()[:, 0]
        vpred_last = self._critic(ob_last).detach().cpu().numpy()[:, 0]
        vpred = np.append(vpred, vpred_last)
        assert len(vpred) == T + 1

        if hasattr(self, "_predict_reward"):
            ob = rollouts["ob"]
            ob_next = rollouts["ob_next"]
            ac = rollouts["ac"]
            rew_il = self._predict_reward(ob, ob_next, ac)
            rew = (1 - self._cfg.gail_env_reward) * rew_il[
                :T
            ] + self._cfg.gail_env_reward * np.array(rew) * self._cfg.reward_scale
            assert rew.shape == (T,)

        adv = np.empty((T,), "float32")
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = (
                rew[t] + self._cfg.rl_discount * vpred[t + 1] * nonterminal - vpred[t]
            )
            adv[t] = lastgaelam = (
                delta
                + self._cfg.rl_discount
                * self._cfg.gae_lambda
                * nonterminal
                * lastgaelam
            )

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        if self._cfg.advantage_norm:
            rollouts["adv"] = ((adv - adv.mean()) / (adv.std() + 1e-5)).tolist()
        else:
            rollouts["adv"] = adv.tolist()

        rollouts["ret"] = ret.tolist()

    def is_off_policy(self):
        return False

    def state_dict(self):
        return {
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "critic_state_dict" not in ckpt:
            # BC initialization
            Logger.warning("Load only actor from BC initialization")
            self._actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
            self.to(self._device)
            self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
            return

        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self.to(self._device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._device)
        optimizer_cuda(self._critic_optim, self._device)

    def update(self):
        train_info = Info()

        copy_network(self._old_actor, self._actor)

        num_batches = self._cfg.train_every // self._cfg.batch_size
        assert num_batches > 0

        for _ in range(self._cfg.ppo_epoch):
            # self._buffer.clear()
            # self._compute_gae(self._rollouts)
            # self._buffer.store_episode(self._rollouts)

            for _ in range(num_batches):
                transitions = self._buffer.sample(self._cfg.batch_size)
                _train_info = self._update_network(transitions)
                train_info.add(_train_info)

        self._buffer.clear()

        Logger.info(
            "Actor lr %f, Critic lr %f, PPO Clip Frac %f",
            self._actor_lr_scheduler.get_last_lr()[0],
            self._critic_lr_scheduler.get_last_lr()[0],
            np.mean(train_info["ppo_clip_frac"]),
        )

        self._actor_lr_scheduler.step()
        self._critic_lr_scheduler.step()

        return mpi_average(train_info.get_dict(only_scalar=True))

    def _update_actor(self, o, ac, adv):
        info = Info()

        _, log_pi, ent = self._actor.act(o, ac=ac, return_log_prob=True)
        _, old_log_pi, _ = self._old_actor.act(o, ac=ac, return_log_prob=True)
        if old_log_pi.min() < -100:
            Logger.error("Sampling an action with a probability of 1e-100")
            import ipdb

            ipdb.set_trace()

        # the actor loss
        entropy_loss = -self._cfg.entropy_loss_coeff * ent.mean()
        ratio = torch.exp(log_pi - old_log_pi)

        surr1 = ratio * adv
        surr2 = (
            torch.clamp(ratio, 1.0 - self._cfg.ppo_clip, 1.0 + self._cfg.ppo_clip) * adv
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        if actor_loss.isnan() or ratio.isnan().any() or adv.isnan().any():
            Logger.error("actor loss is NaN")
            import ipdb

            ipdb.set_trace()

        ppo_clip_frac = (
            torch.gt(torch.abs(ratio - 1.0), self._cfg.ppo_clip).float().mean()
        )

        info["ppo_clip_frac"] = ppo_clip_frac.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["entropy"] = ent.mean().cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), self._cfg.max_grad_norm
            )
        sync_grad(self._actor, self._device)
        self._actor_optim.step()

        # include info from policy
        info.add(self._actor.info)

        return info

    def _update_critic(self, o, ret):
        info = Info()

        # the q loss
        value_pred = self._critic(o)
        value_loss = self._cfg.value_loss_coeff * (ret - value_pred).pow(2).mean()

        # update the critic
        self._critic_optim.zero_grad()
        value_loss.backward()
        if self._cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._critic.parameters(), self._cfg.max_grad_norm
            )
        sync_grad(self._critic, self._device)
        self._critic_optim.step()

        info["value_target"] = ret.mean().cpu().item()
        info["value_predicted"] = value_pred.mean().cpu().item()
        info["value_loss"] = value_loss.cpu().item()

        return info

    def _update_network(self, transitions):
        info = Info()

        # pre-process observations
        o = transitions["ob"]
        o = self.normalize(o)

        bs = len(transitions["done"])
        _to_tensor = lambda x: to_tensor(x, self._device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions["ac"])
        ret = _to_tensor(transitions["ret"]).reshape(bs, 1)
        adv = _to_tensor(transitions["adv"]).reshape(bs, 1)

        self._update_iter += 1

        critic_train_info = self._update_critic(o, ret)
        info.add(critic_train_info)

        if self._update_iter % self._cfg.actor_update_freq == 0:
            actor_train_info = self._update_actor(o, ac, adv)
            info.add(actor_train_info)

        return info
