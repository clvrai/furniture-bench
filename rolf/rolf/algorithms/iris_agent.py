from collections import OrderedDict
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym.spaces
from torch.optim.lr_scheduler import StepLR

from .base_agent import BaseAgent
from .rollout import RolloutRunner
from ..algorithms.dataset import ReplayBufferEpisode, SeqSampler
from ..networks.utils import MLP, get_activation
from ..networks.distributions import Normal, TanhNormal, normal_kl
from ..utils import Logger, Info
from ..utils.pytorch import optimizer_cuda, count_parameters, to_tensor


class IRISAgent(BaseAgent):

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._epoch = 0
        self._skill_t = 0

        state_dim = gym.spaces.flatdim(ob_space)
        ac_dim = gym.spaces.flatdim(ac_space)
        self.actor = Actor(cfg, state_dim * 2, ac_dim)
        self.encoder = Encoder(cfg.encoder2, state_dim * 2, cfg.vae_dim)
        self.decoder = Decoder(cfg.decoder, state_dim, cfg.vae_dim, state_dim)

        self.to(self._device)
        self._optim = optim.Adam(self.parameters(), lr=cfg.bc_lr)

        self._scheduler = StepLR(
            self._optim,
            step_size=cfg.max_global_step // 5,
            gamma=0.5,
        )

        self._log_creation()

        buffer_keys = ["ob", "ac", "done"]
        sampler = SeqSampler(cfg.skill_horizon + 1)
        self._buffer = ReplayBufferEpisode(
            buffer_keys, None, sampler.sample_func_tensor, cfg.precision
        )
        self._val_buffer = ReplayBufferEpisode(
            buffer_keys, None, sampler.sample_func_tensor, cfg.precision
        )

        # Load data
        Logger.info("Load data")
        with open(cfg.data_path, "rb") as f:
            dataset = pickle.load(f)

        clip_to_eps = True
        if clip_to_eps:
            lim = 1 - 1e-5
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (np.linalg.norm(dataset["observations"][i + 1]['robot_state'] -
                               dataset["next_observations"][i]['robot_state']) > 1e-6
                    or dataset["terminals"][i] == 1.0):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        for i in range(len(dones_float)):
            dataset["observations"][i] = gym.spaces.flatten(ob_space, dataset["observations"][i])

        l = len(dones_float)
        e = int(l * 0.95)
        new_d = dict(ob=dataset["observations"][:e], ac=dataset["actions"][:e], done=dones_float[:e])
        self._buffer.store_episode(new_d, False)
        new_d = dict(ob=dataset["observations"][e:], ac=dataset["actions"][e:], done=dones_float[e:])
        self._val_buffer.store_episode(new_d, False)

    def _log_creation(self):
        Logger.info("Creating an IRIS agent")
        Logger.info("The actor has %d parameters", count_parameters(self.actor))
        Logger.info("The encoder has %d parameters", count_parameters(self.encoder))
        Logger.info("The decoder has %d parameters", count_parameters(self.decoder))

    def get_runner(self, cfg, env, env_eval):
        return RolloutRunner(cfg, None, env_eval, self)

    def is_off_policy(self):
        return False

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optim": self._optim.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.actor.load_state_dict(ckpt["actor"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self._ob_norm.load_state_dict(ckpt["ob_norm"])
        self.to(self._device)

        self._optim.load_state_dict(ckpt["optim"])
        self._optim.param_groups[0]['capturable'] = True
        optimizer_cuda(self._optim, self._device)

    def act(self, ob, state, is_train=True):
        ob = gym.spaces.flatten(self._ob_space, ob)
        ob = np.expand_dims(ob, axis=0)
        ob = to_tensor(ob, self._device)

        if state is None or self._skill_t == 0:
            self._skill_t = self._cfg.skill_horizon
            goal = self.decoder.sample_goal(ob)
            h = c = None
        else:
            goal, h, c = state

        ac, (h, c) = self.actor.act(ob, goal, h, c)
        ac = ac.detach().cpu().numpy().squeeze(0)
        ac = gym.spaces.unflatten(self._ac_space, ac)

        self._skill_t -= 1
        return ac, (goal, h, c)

    def update(self):
        train_info = Info()
        for _ in range(self._cfg.train_iter):
            batch = self._buffer.sample(self._cfg.batch_size)
            _train_info = self._update_network(batch)
            train_info.add(_train_info)
        self._scheduler.step()
        return train_info

    def evaluate(self):
        batch = self._val_buffer.sample(self._cfg.batch_size)
        return self._update_network(batch, train=False)

    def _update_network(self, batch, train=True):
        info = Info()
        cfg = self._cfg
        B, H = cfg.batch_size, cfg.skill_horizon
        mse = nn.MSELoss()

        o, ac = batch["ob"], batch["ac"]
        g = o[:, -1, :]

        latent_dist = self.encoder(o[:, 0, :], g)
        z = latent_dist.sample()
        g_pred = self.decoder(o[:, 0, :], z)

        vae_kl_div_loss = cfg.vae_kl_coef * normal_kl(latent_dist).mean()
        goal_recon_loss = cfg.goal_recon_coef * mse(g_pred, g)

        pred_ac = self.actor(o[:, :-1, :], g)
        bc_loss = cfg.bc_coef * mse(pred_ac, ac[:, :-1, :])

        loss = vae_kl_div_loss + goal_recon_loss + bc_loss

        info["vae_kl_div_loss"] = vae_kl_div_loss.cpu().item()
        info["goal_recon_loss"] = (goal_recon_loss / cfg.goal_recon_coef).cpu().item()
        info["actor_loss"] = (bc_loss / cfg.bc_coef).cpu().item()

        if train:
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

        return info


class Encoder(nn.Module):
    def __init__(self, cfg, input_dim, vae_dim):
        super().__init__()
        act = get_activation(cfg.dense_act)
        self.fc = MLP(input_dim, vae_dim * 2, cfg.hidden_dims, act)

        init_std = 5
        self._raw_init_std = np.log(np.exp(init_std) - 1)
        self._min_std = 1e-4
        self._max_std = 10

    def forward(self, *input):
        input = torch.cat(input, -1)
        out = self.fc(input)
        mean, std = out.chunk(2, dim=-1)
        std = F.softplus(std + self._raw_init_std) + self._min_std
        std = torch.clamp(std, max=self._max_std)
        return Normal(mean, std, event_dim=1)

    def sample(self, input):
        return self.forward(input).sample()


class Decoder(nn.Module):
    def __init__(self, cfg, state_dim, latent_dim, output_dim):
        super().__init__()
        act = get_activation(cfg.dense_act)
        self.fc = MLP(state_dim + latent_dim, output_dim, cfg.hidden_dims, act)

        self._latent_dim = latent_dim

    def forward(self, *input):
        input = torch.cat(input, -1)
        out = self.fc(input)
        return out

    def sample_goal(self, state):
        with torch.no_grad():
            shape = list(state.shape[:-1]) + [self._latent_dim]
            latent = torch.normal(torch.zeros(shape), torch.ones(shape)).cuda()
            return self.forward(state, latent)


class Actor(nn.Module):
    def __init__(self, cfg, input_dim, ac_dim):
        super().__init__()
        act = get_activation(cfg.policy_activation)

        self.embed = MLP(input_dim, cfg.lstm_size, [1024, 1024, cfg.lstm_size], act)
        self.lstm = nn.LSTM(cfg.lstm_size, cfg.lstm_size, batch_first=True)
        self.output = MLP(cfg.lstm_size, ac_dim, [512, 256], act)

    def forward(self, states, goals):
        batch_size, seq_len = states.shape[:2]

        goals = goals.view(batch_size, 1, -1).expand(-1, seq_len, -1)
        # goals = torch.repeat_interleave(goals[:, 0, :][:, None], repeats=seq_len, dim=1)
        gc_states = self.embed(torch.cat([states, goals], dim=-1))

        lstm_out, _ = self.lstm(gc_states.view(batch_size, seq_len, -1))

        # actions = self.output(lstm_out.reshape(batch_size * seq_len, -1))
        actions = self.output(lstm_out)

        return actions

    def act(self, state, goal, h, c):
        gc_state = self.embed(torch.cat([state, goal], dim=-1)).view(1, 1, -1)
        if h is not None:
            lstm_out, (h, c) = self.lstm(gc_state, (h, c))
        else:
            lstm_out, (h, c) = self.lstm(gc_state)

        action = self.output(lstm_out.reshape(1, -1))

        return action, (h, c)
