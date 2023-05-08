from collections import OrderedDict, deque

import numpy as np
import gym.spaces

from .mpi import mpi_sum
from .logger import Logger


class SubNormalizer:
    def __init__(self, size, eps=1e-6, clip_range=np.inf, clip_obs=np.inf):
        if isinstance(size, list):
            self.size = size
        else:
            self.size = [size]
        self.eps = eps
        self.clip_range = clip_range
        self.clip_obs = float(clip_obs)  # FIX: str(inf) to float(inf) in hydra
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def _clip(self, v):
        return np.clip(v, -self.clip_obs, self.clip_obs)

    # update the parameters of the normalizer
    def update(self, v):
        v = self._clip(v)
        v = v.reshape([-1] + self.size)

        if not isinstance(v, np.ndarray):
            v = v.detach().numpy()
        # do the computing
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = mpi_sum(local_sum)
        local_sumsq[...] = mpi_sum(local_sumsq)
        local_count[...] = mpi_sum(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # synrc the stats
        sync_sum, sync_sumsq, sync_count = self.sync(
            local_sum, local_sumsq, local_count
        )
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(
            np.maximum(
                np.square(self.eps),
                (self.total_sumsq / self.total_count)
                - np.square(self.total_sum / self.total_count),
            )
        )

    # normalize the observation
    def normalize(self, v):
        v = self._clip(v)
        return np.clip((v - self.mean) / (self.std), -self.clip_range, self.clip_range)

    def sample(self, alpha=1.0):
        v = self.mean + np.random.normal(size=self.size) * self.std * alpha
        if np.any(np.isnan(v)):
            Logger.error("Sample NaN")
            Logger.error(f"Mean {self.mean}")
            Logger.error(f"Std {self.std}")
            Logger.error(f"Sample {v}")
            raise ValueError
        return v

    def state_dict(self):
        return {
            "sum": self.total_sum,
            "sumsq": self.total_sumsq,
            "count": self.total_count,
        }

    def load_state_dict(self, state_dict):
        self.total_sum = state_dict["sum"]
        self.total_sumsq = state_dict["sumsq"]
        self.total_count = state_dict["count"]
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(
            np.maximum(
                np.square(self.eps),
                (self.total_sumsq / self.total_count)
                - np.square(self.total_sum / self.total_count),
            )
        )


class Normalizer(object):
    def __init__(self, shape, eps=1e-6, clip_range=np.inf, clip_obs=np.inf):
        if isinstance(shape, gym.spaces.Dict):
            self._shape= {}
            for k, v in shape.spaces.items():
                if v.shape is not None:
                    self._shape[k] = list(v.shape)
            # self._shape = {k: list(v.shape) for k, v in shape.spaces.items()}
        elif isinstance(shape, dict):
            self._shape = shape
        else:
            self._shape = {"": shape}
        Logger.info(f"Initialize ob_norm with shape: {self._shape}")

        self._keys = sorted(self._shape.keys())

        self.sub_norm = {}
        for key in self._keys:
            self.sub_norm[key] = SubNormalizer(
                self._shape[key], eps, clip_range, clip_obs
            )

    # update the parameters of the normalizer
    def update(self, v):
        if isinstance(v, (list, deque)):
            if isinstance(v[0], dict):
                v = OrderedDict(
                    [(k, np.asarray([x[k] for x in v])) for k in self._keys]
                )
            else:
                v = np.asarray(v)

        if isinstance(v, dict):
            for k, v_ in v.items():
                if k in self._keys:
                    self.sub_norm[k].update(v_)
        else:
            self.sub_norm[""].update(v)

    def recompute_stats(self):
        for k in self._keys:
            self.sub_norm[k].recompute_stats()

    # normalize the observation
    def _normalize(self, v):
        if not isinstance(v, dict):
            return self.sub_norm[""].normalize(v)
        return OrderedDict(
            [
                (k, self.sub_norm[k].normalize(v_))
                for k, v_ in v.items()
                if k in self._keys
            ]
        )

    def normalize(self, v):
        if isinstance(v, list):
            return [self._normalize(x) for x in v]
        else:
            return self._normalize(v)

    def sample(self, alpha=1.0):
        return OrderedDict([(k, self.sub_norm[k].sample(alpha)) for k in self._keys])

    def state_dict(self):
        return OrderedDict([(k, self.sub_norm[k].state_dict()) for k in self._keys])

    def load_state_dict(self, state_dict):
        for k in self._keys:
            self.sub_norm[k].load_state_dict(state_dict[k])
