# Dreamer code reference:
# https://github.com/danijar/dreamer/blob/master/dreamer.py
# TODO: pcont is not implemented yet

import numpy as np
import torch
import gym.spaces

from .base_agent import BaseAgent
from .dataset import ReplayBufferEpisode, SeqSampler
from .dreamer_rollout import DreamerRolloutRunner
from ..networks.dreamer import DreamerModel, DenseDecoder1, ActionDecoder
from ..utils import Logger, Once, Info, StopWatch
from ..utils.pytorch import optimizer_cuda, count_parameters
from ..utils.pytorch import to_tensor, RequiresGrad, AdamAMP
from ..utils.dreamer import static_scan, lambda_return


class DreamerAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._ac_dim = ac_dim = gym.spaces.flatdim(ac_space)
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        state_dim = cfg.deter_dim + cfg.stoch_dim

        # Build up networks
        self._model = DreamerModel(cfg, ob_space, ac_dim, self._dtype)
        self._actor = ActionDecoder(
            state_dim, ac_dim, [cfg.num_units] * 4, cfg.dense_act
        )
        self._critic = DenseDecoder1(state_dim, 1, [cfg.num_units] * 3, cfg.dense_act)
        self.to(self._device)

        # Optimizers
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self._model_optim = adam_amp(self._model, cfg.model_lr)
        self._actor_optim = adam_amp(self._actor, cfg.actor_lr)
        self._critic_optim = adam_amp(self._critic, cfg.critic_lr)

        # Per-episode replay buffer
        sampler = SeqSampler(cfg.batch_length)
        buffer_keys = ["ob", "ac", "rew", "done"]
        self._buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func, cfg.precision
        )

        self._log_creation()

        # Freeze modules. Only updated modules will be unfrozen.
        self.requires_grad_(False)

    @property
    def ac_space(self):
        return self._ac_space

    def _log_creation(self):
        Logger.info("Creating a Dreamer agent")
        Logger.info(f"The actor has {count_parameters(self._actor)} parameters")
        Logger.info(f"The critic has {count_parameters(self._critic)} parameters")
        Logger.info(f"The model has {count_parameters(self._model)} parameters")

    @torch.no_grad()
    def act(self, ob, state, is_train=True):
        """Returns action and the actor's activation given an observation `ob`."""
        ob = ob.copy()
        for k, v in ob.items():
            ob[k] = np.expand_dims(v, axis=0).copy()

        self._model.eval()
        self._actor.eval()
        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            ob = to_tensor(ob, self._device, self._dtype)
            ac, state_next = self._policy(ob, state, is_train)
            ac = ac.cpu().numpy().squeeze(0)
            ac = gym.spaces.unflatten(self._ac_space, ac)
        self._model.train()
        self._actor.train()

        return ac, state_next

    def get_runner(self, cfg, env, env_eval):
        """Returns rollout runner."""
        return DreamerRolloutRunner(cfg, env, env_eval, self)

    def is_off_policy(self):
        return True

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts, include_last_ob=False)

    def state_dict(self):
        return {
            "model_state_dict": self._model.state_dict(),
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "model_optim_state_dict": self._model_optim.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        self.to(self._device)

        self._model_optim.load_state_dict(ckpt["model_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        optimizer_cuda(self._model_optim, self._device)
        optimizer_cuda(self._actor_optim, self._device)
        optimizer_cuda(self._critic_optim, self._device)

    def update(self):
        train_info = Info()
        log_once = Once()
        sw_data = StopWatch()
        sw_train = StopWatch()
        for _ in range(self._cfg.train_steps):
            sw_data.start()
            batch = self._buffer.sample(self._cfg.batch_size)
            sw_data.stop()

            sw_train.start()
            _train_info = self._update_network(batch, log_image=log_once())
            train_info.add(_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")
        # return train_info.get_dict()

        info = train_info.get_dict()
        Logger.info(
            f"model_grad: {info['model_grad_norm']:.1f} / actor_grad: {info['actor_grad_norm']:.1f} / critic_grad: {info['critic_grad_norm']:.1f} / model_loss: {info['model_loss']:.1f} / actor_loss: {info['actor_loss']:.1f} / critic_loss: {info['critic_loss']:.1f} / prior_ent: {info['prior_entropy']:.1f} / post_ent: {info['posterior_entropy']:.1f} / reward_loss: {info['reward_loss']:.1f} / div: {info['kl_loss']:.1f} / actor_ent: {info['actor_entropy']:.1f}"
        )
        return info

    def _update_network(self, batch, log_image=False):
        info = Info()

        o = to_tensor(batch["ob"], self._device, self._dtype)
        ac = to_tensor(batch["ac"], self._device, self._dtype)
        rew = to_tensor(batch["rew"], self._device, self._dtype)
        o = self.preprocess(o)

        # Compute model loss
        with RequiresGrad(self._model):
            with torch.autocast(self._cfg.device, enabled=self._use_amp):
                embed = self._model.encoder(o)
                post, prior = self._model.dynamics.observe(embed, ac)
                feat = self._model.dynamics.get_feat(post)

                ob_pred = self._model.decoder(feat)
                recon_losses = {k: -ob_pred[k].log_prob(v).mean() for k, v in o.items()}
                recon_loss = sum(recon_losses.values())

                reward_pred = self._model.reward(feat)
                reward_loss = -reward_pred.log_prob(rew.unsqueeze(-1)).mean()

                prior_dist = self._model.dynamics.get_dist(prior)
                post_dist = self._model.dynamics.get_dist(post)

                # Clipping KL divergence after taking mean (from official code)
                div = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
                div_clipped = torch.clamp(div, min=self._cfg.free_nats)
                model_loss = self._cfg.kl_scale * div_clipped + recon_loss + reward_loss
            model_grad_norm = self._model_optim.step(model_loss)

        # Compute actor loss with imaginary rollout
        with RequiresGrad(self._actor):
            with torch.autocast(self._cfg.device, enabled=self._use_amp):
                post = {k: v.detach() for k, v in post.items()}
                imagine_feat = self._imagine_ahead(post)
                imagine_reward = (
                    self._model.reward(imagine_feat).mode().squeeze(-1).float()
                )
                imagine_value = self._critic(imagine_feat).mode().squeeze(-1).float()
                pcont = self._cfg.rl_discount * torch.ones_like(imagine_reward)
                imagine_return = lambda_return(
                    imagine_reward[:-1],
                    imagine_value[:-1],
                    pcont[:-1],
                    bootstrap=imagine_value[-1],
                    lambda_=self._cfg.gae_lambda,
                )
                with torch.no_grad():
                    discount = torch.cumprod(
                        torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0
                    )
                actor_loss = -(discount * imagine_return).mean()
            actor_grad_norm = self._actor_optim.step(actor_loss)

        # Compute critic loss
        with RequiresGrad(self._critic):
            with torch.autocast(self._cfg.device, enabled=self._use_amp):
                value_pred = self._critic(imagine_feat.detach()[:-1])
                target = imagine_return.detach().unsqueeze(-1)
                critic_loss = -(discount * value_pred.log_prob(target)).mean()
            critic_grad_norm = self._critic_optim.step(critic_loss)

        # Log scalar
        for k, v in recon_losses.items():
            info[f"recon_loss_{k}"] = v.item()
        info["reward_loss"] = reward_loss.item()
        info["prior_entropy"] = prior_dist.entropy().mean().item()
        info["posterior_entropy"] = post_dist.entropy().mean().item()
        info["kl_loss"] = div_clipped.item()
        info["model_loss"] = model_loss.item()
        info["actor_loss"] = actor_loss.item()
        info["critic_loss"] = critic_loss.item()
        info["value_target"] = imagine_return.mean().item()
        info["value_predicted"] = value_pred.mode().mean().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["critic_grad_norm"] = critic_grad_norm.item()

        if log_image:
            with torch.no_grad(), torch.autocast(
                self._cfg.device, enabled=self._use_amp
            ):
                info["actor_entropy"] = self._actor(feat).entropy().mean().item()

                # 5 timesteps for each of 4 samples
                init, _ = self._model.dynamics.observe(embed[:4, :5], ac[:4, :5])
                init = {k: v[:, -1] for k, v in init.items()}
                prior = self._model.dynamics.imagine(ac[:4, 5:], init)
                openloop = self._model.decoder(
                    self._model.dynamics.get_feat(prior)
                ).mode()
                for k, v in o.items():
                    if len(v.shape) != 5:
                        continue
                    truth = o[k][:4] + 0.5
                    recon = ob_pred[k].mode()[:4]
                    model = torch.cat([recon[:, :5] + 0.5, openloop[k] + 0.5], 1)
                    error = (model - truth + 1) / 2
                    openloop = torch.cat([truth, model, error], 2)
                    img = openloop.detach().cpu().numpy() * 255
                    info[f"recon_{k}"] = img.transpose(0, 1, 4, 2, 3).astype(np.uint8)

        return info.get_dict()

    def _imagine_ahead(self, post):
        """Computes imagination rollouts.
        Args:
            post: BxTx(`stoch_dim` + `deter_dim`) stochastic states.
        """
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        policy = lambda state: self._actor.act(
            self._model.dynamics.get_feat(state).detach()
        )
        imagine_states = static_scan(
            lambda prev, _: self._model.dynamics.imagine_step(prev, policy(prev)),
            [torch.arange(self._cfg.horizon)],
            start,
        )
        imagine_feat = self._model.dynamics.get_feat(imagine_states)
        return imagine_feat

    def _policy(self, ob, state, is_train):
        """Computes actions given `ob` and `state`.

        Args:
            ob: list of B observations (tensors)
            state: (previous_latent_state, previous_action)
        """
        latent, action = state or self.initial_state(ob)
        embed = self._model.encoder(self.preprocess(ob))
        latent = self._model.dynamics.obs_step(latent, action, embed)
        feat = self._model.dynamics.get_feat(latent)
        action = self._actor.act(feat, deterministic=not is_train)
        if is_train:
            action = action + torch.randn_like(action) * self._cfg.expl_noise
            action = torch.clamp(action, -1, 1)
        state = (latent, action)
        return action, state

    def initial_state(self, ob):
        batch_size = len(list(ob.values())[0])
        latent = self._model.initial(batch_size)
        action = torch.zeros(
            [batch_size, self._ac_dim], dtype=self._dtype, device=self._device
        )
        return latent, action

    def preprocess(self, ob):
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                ob[k] = ob[k] / 255.0 - 0.5
        return ob
