from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces

from .utils import MLP, get_activation
from .distributions import Normal, TanhNormal, SampleDist, MixedDistribution

from ..utils import Logger
from ..utils.dreamer import static_scan


class DreamerModel(nn.Module):
    def __init__(self, cfg, ob_space, ac_dim: float, dtype):
        super().__init__()
        self._cfg = cfg
        self._ob_space = ob_space
        self._dtpe = dtype

        self.encoder = Encoder(cfg.encoder, ob_space)

        embed_dim = self.encoder.output_dim
        self.dynamics = RSSM(
            embed_dim,
            ac_dim,
            cfg.stoch_dim,
            cfg.deter_dim,
            cfg.deter_dim,
            cfg.dense_act,
            dtype,
            cfg.device,
        )

        state_dim = cfg.deter_dim + cfg.stoch_dim
        self.decoder = Decoder(cfg.decoder, state_dim, ob_space)
        self.reward = DenseDecoder1(state_dim, 1, [cfg.num_units] * 2, cfg.dense_act)

    def initial(self, batch_size):
        return self.dynamics.initial(batch_size)


class Encoder(nn.Module):
    def __init__(self, cfg, ob_space):
        super().__init__()
        self._ob_space = ob_space
        self.encoders = nn.ModuleDict()
        self.output_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) == 3:
                self.encoders[k] = ConvEncoder(
                    cfg.image_shape,
                    cfg.kernel_size,
                    cfg.stride,
                    cfg.conv_dim,
                    cfg.cnn_act,
                )
            elif len(v.shape) == 1:
                self.encoders[k] = DenseEncoder(
                    gym.spaces.flatdim(v),
                    cfg.embed_dim,
                    cfg.hidden_dims,
                    cfg.dense_act,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")
            self.output_dim += self.encoders[k].output_dim

    def forward(self, ob):
        if not isinstance(ob, dict):
            ob = {"ob": ob}
        embeddings = [self.encoders[k](v) for k, v in ob.items()]
        return torch.cat(embeddings, -1)


class DenseEncoder(nn.Module):
    def __init__(self, shape, embed_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(shape, embed_dim, hidden_dims, activation)
        self.output_dim = embed_dim

    def forward(self, ob):
        return self.fc(ob)


class ConvEncoder(nn.Module):
    def __init__(self, shape, kernel_size, stride, conv_dim, activation):
        super().__init__()
        convs = []
        activation = get_activation(activation)
        h, w, d_prev = shape
        for k, s, d in zip(kernel_size, stride, conv_dim):
            convs.append(nn.Conv2d(d_prev, d, k, s))
            convs.append(activation)
            d_prev = d
            h = int(np.floor((h - k) / s + 1))
            w = int(np.floor((w - k) / s + 1))

        self.convs = nn.Sequential(*convs)
        self.output_dim = h * w * d_prev

    def forward(self, ob):
        shape = list(ob.shape[:-3]) + [-1]
        x = ob.reshape([-1] + list(ob.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        return x.reshape(shape)


class Decoder(nn.Module):
    def __init__(self, cfg, state_dim, ob_space):
        super().__init__()
        self._ob_space = ob_space
        self.decoders = nn.ModuleDict()
        for k, v in ob_space.spaces.items():
            if len(v.shape) == 3:
                self.decoders[k] = ConvDecoder(
                    state_dim,
                    cfg.image_shape,
                    cfg.kernel_size,
                    cfg.stride,
                    cfg.conv_dim,
                    cfg.cnn_act,
                )
            elif len(v.shape) == 1:
                self.decoders[k] = DenseDecoder(
                    state_dim,
                    gym.spaces.flatdim(v),
                    cfg.hidden_dims,
                    cfg.dense_act,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")

    def forward(self, feat):
        return MixedDistribution(
            OrderedDict([(k, self.decoders[k](feat)) for k in self._ob_space.spaces])
        )


class ConvDecoder(nn.Module):
    """CNN decoder returns normal distribution of prediction with std 1."""

    def __init__(self, input_dim, shape, kernel_size, stride, conv_dim, activation):
        super().__init__()
        self._shape = list(shape)
        self._conv_dim = conv_dim

        self.fc = MLP(input_dim, conv_dim[0], [], None)

        d_prev = conv_dim[0]
        conv_dim = conv_dim + [shape[-1]]
        activation = get_activation(activation)
        deconvs = []
        for k, s, d in zip(kernel_size, stride, conv_dim[1:]):
            deconvs.append(nn.ConvTranspose2d(d_prev, d, k, s))
            deconvs.append(activation)
            d_prev = d
        deconvs.pop()
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, feat):
        shape = list(feat.shape[:-1]) + self._shape
        x = self.fc(feat)
        x = x.reshape([-1, self._conv_dim[0], 1, 1])
        x = self.deconvs(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(shape)
        return Normal(x, 1, event_dim=3)


class DenseDecoder(nn.Module):
    """MLP decoder returns normal distribution of prediction."""

    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(input_dim, output_dim * 2, hidden_dims, activation)

    def forward(self, feat):
        mean, std = self.fc(feat).chunk(2, dim=-1)
        std = (std.tanh() + 1) * 0.7 + 0.1  # [0.1, 1.5]
        return Normal(mean, std, event_dim=1)


class DenseDecoder1(nn.Module):
    """MLP decoder returns normal distribution of prediction with std 1."""

    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(self, feat):
        return Normal(self.fc(feat), 1, event_dim=1)


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim,
        ac_dim,
        stoch_dim,
        deter_dim,
        hidden_dim,
        activation,
        dtype,
        device,
    ):
        """Dynamics model.

        Args:
            embed_dim: size of observation embedding.
            ac_dim: size of action, |a|.
            stoch_dim: size of stochastic latent state, |s|.
            deter_dim: size of deterministic latent state, |h|.
            hidden_dim: size of MLP hidden layers
            dtype: data type for initial states
            device: device for torch tensors
        """
        super().__init__()
        self._stoch_dim = stoch_dim
        self._deter_dim = deter_dim
        self._activation = get_activation(activation)
        self._dtype = dtype
        self._device = device

        self.cell = nn.GRUCell(hidden_dim, deter_dim)
        self.deter_fc = MLP(stoch_dim + ac_dim, hidden_dim, [], activation)
        self.obs_fc = MLP(
            deter_dim + embed_dim, 2 * stoch_dim, [hidden_dim], activation
        )
        self.imagine_fc = MLP(deter_dim, 2 * stoch_dim, [hidden_dim], activation)

    def initial(self, batch_size):
        zeros = lambda s: torch.zeros(
            [batch_size, s], dtype=self._dtype, device=self._device
        )
        return dict(
            mean=zeros(self._stoch_dim),
            std=zeros(self._stoch_dim),
            stoch=zeros(self._stoch_dim),
            deter=zeros(self._deter_dim),
        )

    def observe(self, embed, action, state=None):
        """Computes state posterior and prior on training batch.

        Args:
            embed: Observation embedding of size BxTx`num_units`.
            action: Actions of size BxTx`action_dim`.
            state: (Optional) Stochastic and deterministic states of size Bx(`stoch_dim`+`deter_dim`).
        """
        if state is None:
            state = self.initial(action.shape[0])
        embed, action = embed.transpose(0, 1), action.transpose(0, 1)
        post, prior = static_scan(
            lambda prev, inputs: self.both_step(prev[0], inputs[0], inputs[1]),
            (action, embed),
            (state, state),
        )
        post = {k: v.transpose(0, 1) for k, v in post.items()}
        prior = {k: v.transpose(0, 1) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        """Imaginary rollouts for debugging.

        Args:
            action: Actions of size BxTx`action_dim`.
            state: (Optional) Stochastic and deterministic states of size Bx`stoch_dim` and Bx`deter_dim`.
        """
        if state is None:
            state = self.initial(action.shape[0])
        action = action.transpose(0, 1)
        prior = static_scan(
            lambda prev, inputs: self.imagine_step(prev, inputs[0]), [action], state
        )
        prior = {k: v.transpose(0, 1) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        return torch.cat([state["stoch"], state["deter"]], -1)

    def get_dist(self, state):
        return Normal(state["mean"], state["std"], event_dim=1)

    def get_state(self, x, deter):
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({"mean": mean, "std": std}).rsample()
        return {"mean": mean, "std": std, "stoch": stoch, "deter": deter}

    def deter_step(self, prev_state, prev_action):
        """Deterministic state model, h=f(h',s',a').

        Args:
            prev_state: previous deterministic state, h', and stochastic state, s'
            prev_action: previous action, a'
        """
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self.deter_fc(x)
        x = self._activation(x)
        deter = self.cell(x, prev_state["deter"])
        return deter

    def imagine_step(self, prev_state, prev_action):
        """State prior without observation, p(s|h), where h=f(h',s',a').

        Args:
            prev_state: previous deterministic state, h', and stochastic state, s'
            prev_action: previous action, a'
        """
        deter = self.deter_step(prev_state, prev_action)
        x = self.imagine_fc(deter)
        prior = self.get_state(x, deter)
        return prior

    def obs_step(self, prev_state, prev_action, embed):
        """State posterior with observation, q(s|h,o), where h=f(h',s',a').

        Args:
            prev_state: previous deterministic state, h', and stochastic state, s'
            prev_action: previous action, a'
            embed: current observation, o
        """
        deter = self.deter_step(prev_state, prev_action)
        x = torch.cat([deter, embed], -1)
        x = self.obs_fc(x)
        post = self.get_state(x, deter)
        return post

    def both_step(self, prev_state, prev_action, embed):
        """Returns both state posterior (`obs_step`) and prior (`imagine_step`).

        Args:
            prev_state: previous deterministic state, h', and stochastic state, s'
            prev_action: previous action, a'
            embed: current observation, o
        """
        deter = self.deter_step(prev_state, prev_action)
        x = self.imagine_fc(deter)
        prior = self.get_state(x, deter)

        x = torch.cat([deter, embed], -1)
        x = self.obs_fc(x)
        post = self.get_state(x, deter)
        return post, prior


class ActionDecoder(nn.Module):
    """MLP decoder returns tanh normal distribution of action."""

    def __init__(self, input_dim, ac_dim, hidden_dims, activation):
        super().__init__()

        init_std = 5
        self._raw_init_std = np.log(np.exp(init_std) - 1)
        self._min_std = 1e-4
        self._mean_scale = 5

        self.fc = MLP(input_dim, ac_dim * 2, hidden_dims, activation)

    def forward(self, state):
        mean, std = self.fc(state).chunk(2, dim=-1)
        mean = self._mean_scale * (mean / self._mean_scale).tanh()
        std = F.softplus(std + self._raw_init_std) + self._min_std
        return SampleDist(TanhNormal(mean, std, event_dim=1))

    def act(self, state, cond=None, deterministic=False, return_dist=False):
        """Samples action for rollout."""
        if cond is not None:
            state = torch.cat([state, cond], -1)
        dist = self.forward(state)
        if return_dist:
            return dist
        action = dist.mode() if deterministic else dist.rsample()
        return action
