from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.distributions


# Identity
class Identity(object):
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return self.mean

    def sample(self):
        return self.mean.detach()

    def rsample(self):
        return self.mean


class TanhIdentity(object):
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return torch.tanh(self.mean)

    def sample(self):
        return torch.tanh(self.mean).detach()

    def rsample(self):
        return torch.tanh(self.mean)


# Categorical
class Categorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_prob(self, actions):
        return super().log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def entropy(self):
        return super().entropy() * 10.0  # scailing

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class Normal(torch.distributions.Independent):
    def __init__(self, mean, std, event_dim=0):
        super().__init__(torch.distributions.Normal(mean, std), event_dim)

    def mode(self):
        return self.mean


class AddBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class DiagGaussian(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._device = torch.device(cfg.device)
        self.logstd = AddBias(torch.zeros(cfg.action_size, device=self._device))

    def forward(self, x):
        logstd = self.logstd(torch.zeros_like(x))
        return Normal(x, logstd.exp(), 1)


# Tanh Normal
class TanhTransform(torch.distributions.TanhTransform):
    def _inverse(self, y):
        """Clamp y for numerical stability."""
        dtype = y.dtype
        y = torch.clamp(y.float(), -0.99999997, 0.99999997)
        x = torch.atanh(y)
        return x.type(dtype)


class TanhNormal_(torch.distributions.transformed_distribution.TransformedDistribution):
    """X ~ Normal(loc, scale).
    Y ~ TanhNormal(loc, scale) = tanh(X).
    """

    def __init__(self, loc, scale):
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, TanhTransform())

    @property
    def mean(self):
        return self.base_dist.mean.tanh()


class TanhNormal(torch.distributions.Independent):
    def __init__(self, mean, std, event_dim=0):
        super().__init__(TanhNormal_(mean, std), event_dim)

    def mode(self):
        return self.mean

    def entropy(self):
        """No analytic form. Instead, use entropy of Normal as proxy."""
        return self.base_dist.base_dist.entropy()


class SampleDist(nn.Module):
    def __init__(self, dist, samples=100):
        super().__init__()
        self.base_dist = dist
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self.base_dist, name)

    @property
    def mean(self):
        samples = self.base_dist.rsample((self._samples,))
        return torch.mean(samples, 0)

    def mode(self):
        samples = self.base_dist.rsample((self._samples,))
        log_prob = self.base_dist.log_prob(samples)
        return samples[torch.argmax(log_prob)]

    def entropy(self):
        samples = self.base_dist.rsample((self._samples,))
        log_prob = self.base_dist.log_prob(samples)
        return -torch.mean(log_prob, 0)


def normal_kl(p, q=None):
    """Computes KL divergence based on base normal dist."""
    if q is None or q == "tanh":
        mean = torch.zeros_like(p.mean)
        std = torch.ones_like(p.mean)
        if q is None:
            q = Normal(mean, std, len(p.event_shape))
        else:
            q = TanhNormal(mean, std, len(p.event_shape))

    # TODO: fix p.base_dist.base_dist for TanhNormal and p.base_dist for Normal
    p_base = torch.distributions.Independent(p.base_dist, len(p.event_shape))
    q_base = torch.distributions.Independent(q.base_dist, len(q.event_shape))
    return torch.distributions.kl.kl_divergence(p_base, q_base)


class MixedDistribution(nn.Module):
    def __init__(self, base_dists):
        super().__init__()
        assert isinstance(base_dists, OrderedDict)
        self.base_dists = base_dists

    def __getitem__(self, key):
        return self.base_dists[key]

    def mode(self):
        return OrderedDict([(k, dist.mode()) for k, dist in self.base_dists.items()])

    def sample(self):
        return OrderedDict([(k, dist.sample()) for k, dist in self.base_dists.items()])

    def rsample(self):
        return OrderedDict([(k, dist.rsample()) for k, dist in self.base_dists.items()])

    def log_prob(self, x):
        assert isinstance(x, dict)
        log_prob = OrderedDict(
            [(k, dist.log_prob(x[k])) for k, dist in self.base_dists.items()]
        )
        return torch.stack(list(log_prob.values()), -1).sum(-1, keepdim=True)

    def entropy(self):
        return sum([dist.entropy() for dist in self.base_dists.values()])
