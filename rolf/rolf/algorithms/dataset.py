from collections import defaultdict, deque
from functools import partial

import numpy as np
import tensorflow as tf
import torch.utils.data

from ..utils import Logger


def _convert(value, precision):
    if isinstance(value, dict):
        return {k: _convert(v, precision) for k, v in value.items()}

    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32}[precision]
    else:
        dtype = value.dtype
    return value.astype(dtype)


class ReplayBuffer(object):
    """Replay buffer to store trainsitions in list (deque)."""

    def __init__(self, keys, buffer_size, sample_func):
        self._capacity = buffer_size
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self.clear()

    @property
    def size(self):
        return self._capacity

    @property
    def last_saved_idx(self):
        return self._last_saved_idx

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._last_saved_idx = -1
        self._buffer = defaultdict(partial(deque, maxlen=self._capacity))

    def store_episode(self, rollout):
        """`rollout` can be any length of transitions."""
        for k in self._keys:
            self._buffer[k].append(rollout[k])

        self._idx += 1
        if self._current_size < self._capacity:
            self._current_size += 1

    def sample(self, batch_size):
        transitions = self._sample_func(self._buffer, batch_size)
        return transitions

    def state_dict(self):
        """Returns new transitions in replay buffer."""
        assert self._idx - self._last_saved_idx - 1 <= self._capacity
        state_dict = {}
        s = (self._last_saved_idx + 1) % self._capacity
        e = (self._idx - 1) % self._capacity
        for k in self._keys:
            state_dict[k] = list(self._buffer[k])
            if s < e:
                state_dict[k] = state_dict[k][s : e + 1]
            else:
                state_dict[k] = state_dict[k][s:] + state_dict[k][: e + 1]
            assert len(state_dict[k]) == self._idx - self._last_saved_idx - 1
        self._last_saved_idx = self._idx - 1
        Logger.info(f"Store {len(state_dict['ac'])} states")
        return state_dict

    def append_state_dict(self, state_dict):
        """Adds transitions to replay buffer."""
        for k in self._keys:
            self._buffer[k].extend(state_dict[k])

        n = len(state_dict["ac"])
        self._last_saved_idx += n
        self._idx += n
        Logger.info(f"Load {n} states")


class ReplayBufferEpisode(object):
    """Replay buffer storing each episode in file."""

    def __init__(self, keys, buffer_size, sample_func, precision=32):
        super().__init__()
        self._capacity = buffer_size
        self._sample_func = sample_func
        self._precision = precision

        # create the buffer to store info
        self._keys = keys
        self.clear()

    @property
    def size(self):
        return self._capacity

    @property
    def buffer(self):
        return self._buffer

    def clear(self):
        self._idx = 0
        self._new_episode = True
        self._last_saved_idx = -1
        self._buffer = deque(maxlen=self._capacity)
        self._rollout = {k: [] for k in self._keys}

    def store_episode(self, rollout, include_last_ob=True):
        """Stores `rollout` into `self._buffer`.

        Args:
            rollout: A dictionary of lists of transitions.
            include_last_ob: Include the last observation. Thus, len(ob) = len(ac) + 1
        """
        rollout_len = len(rollout["ac"])
        for i in range(rollout_len):
            for k in self._keys:
                self._rollout[k].append(rollout[k][i])

            if rollout["done"][i]:
                if include_last_ob:
                    # Add the last observation.
                    self._rollout["ob"].append(rollout["ob_next"][i])

                episode = {}
                for k in self._keys:
                    episode[k] = {}
                    if isinstance(self._rollout[k][0], dict):
                        for sub_key in self._rollout[k][0]:
                            v = np.array([t[sub_key] for t in self._rollout[k]])
                            v = _convert(v, self._precision)
                            episode[k][sub_key] = torch.as_tensor(v)
                    else:
                        v = np.array(self._rollout[k])
                        episode[k] = torch.as_tensor(_convert(v, self._precision))

                self._buffer.append(episode)
                self._idx += 1
                self._rollout = {k: [] for k in self._keys}

    def sample(self, batch_size):
        batch = self._sample_func(self._buffer, batch_size)
        return batch

    def state_dict(self):
        """Returns new transitions in replay buffer."""
        assert self._idx - self._last_saved_idx - 1 <= self._capacity
        state_dict = {}
        s = (self._last_saved_idx + 1) % self._capacity
        e = (self._idx - 1) % self._capacity
        l = list(self._buffer)
        if s <= e:
            state_dict = {"episodes": l[s : e + 1]}
        else:
            state_dict = {"episodes": l[s:] + l[: e + 1]}
        self._last_saved_idx = self._idx - 1
        Logger.info(f"Store {len(state_dict['episodes'])} episodes")
        return state_dict

    def append_state_dict(self, state_dict):
        """Adds transitions to replay buffer."""
        self._buffer.extend(state_dict["episodes"])

        n = len(state_dict["episodes"])
        self._last_saved_idx += n
        self._idx += n
        Logger.info(f"Load {n} episodes")


class RandomSampler(object):
    def __init__(self):
        pass

    def sample_func(self, dataset, batch_size):
        key_len = "ac" if "ac" in dataset else "ob"
        dataset_size = len(dataset[key_len])

        episode_idxs = np.random.randint(0, dataset_size, batch_size)
        t_samples = [
            np.random.randint(len(dataset[key_len][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in dataset.keys():
            transitions[key] = [
                dataset[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions


class SeqSampler(object):
    def __init__(self, seq_len, sample_last_more=False):
        self._seq_len = seq_len
        self._sample_last_more = sample_last_more

    def sample_func(self, dataset, batch_size):
        data = []
        dataset_len = len(dataset)
        for _ in range(batch_size):
            while True:
                episode_idx = np.random.randint(dataset_len)
                episode = dataset[episode_idx].copy()
                episode_len = len(episode["done"])
                if episode_len >= self._seq_len:
                    break

            t = np.random.randint(episode_len - self._seq_len + 1)
            traj = [v[t : t + self._seq_len] for v in tf.nest.flatten(episode)]
            data.append(traj)

        batch = [np.stack([v[i] for v in data]) for i in range(len(traj))]
        return tf.nest.pack_sequence_as(episode, batch)

    def sample_func_tensor(self, dataset, batch_size):
        """Sample batch in tensor."""
        dataset_len = len(dataset)
        episode = dataset[0].copy()
        new_tensor = lambda v, l: torch.empty(
            (batch_size, l, *v.shape[1:]), dtype=v.dtype
        )
        batch = [new_tensor(v, self._seq_len) for v in tf.nest.flatten(episode)]

        for i in range(batch_size):
            while True:
                episode_idx = np.random.randint(dataset_len)
                episode = dataset[episode_idx].copy()
                episode_len = len(episode["done"])
                if episode_len >= self._seq_len:
                    break

            t = np.random.randint(episode_len - self._seq_len + 1)
            for j, v in enumerate(tf.nest.flatten(episode)):
                batch[j][i] = v[t : t + self._seq_len]

        # TODO: change cuda() to to(self._device)
        batch = [b.cuda() for b in batch]
        # batch = [b for b in batch]
        return tf.nest.pack_sequence_as(episode, batch)

    def sample_func_one_more_ob(self, dataset, batch_size):
        """Sample one more `ob` than other items."""
        # 0.18 sec (numpy array) -> 0.08 sec (empty cuda tensor) -> 0.067 sec (cuda at once)
        dataset_len = len(dataset)
        episode = dataset[0].copy()
        ob = episode.pop("ob")
        new_tensor = lambda v, l: torch.empty(
            (batch_size, l, *v.shape[1:]), dtype=v.dtype
        )
        batch_ob = [new_tensor(v, self._seq_len + 1) for v in tf.nest.flatten(ob)]
        batch_ep = [new_tensor(v, self._seq_len) for v in tf.nest.flatten(episode)]

        for i in range(batch_size):
            while True:
                episode_idx = np.random.randint(dataset_len)
                episode = dataset[episode_idx].copy()
                episode_len = len(episode["done"])
                if episode_len >= self._seq_len:
                    break

            if self._sample_last_more:
                t = np.random.randint(episode_len)
                t = np.clip(t, 0, episode_len - self._seq_len)
            else:
                t = np.random.randint(episode_len - self._seq_len + 1)
            ob = episode.pop("ob")
            for j, v in enumerate(tf.nest.flatten(ob)):
                batch_ob[j][i] = v[t : t + self._seq_len + 1]
            for j, v in enumerate(tf.nest.flatten(episode)):
                batch_ep[j][i] = v[t : t + self._seq_len]

        # TODO: change cuda() to to(self._device)
        batch_ob = [b.cuda() for b in batch_ob]
        batch_ep = [b.cuda() for b in batch_ep]

        batch_ob = tf.nest.pack_sequence_as(ob, batch_ob)
        batch_ep = tf.nest.pack_sequence_as(episode, batch_ep)
        batch_ep.update(dict(ob=batch_ob))

        return batch_ep


class HERSampler(object):
    def __init__(self, replay_strategy, replace_future, reward_func=None):
        self.replay_strategy = replay_strategy
        if self.replay_strategy == "future":
            self.future_p = replace_future
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, dataset, batch_size):
        key_len = "ac" if "ac" in dataset else "ob"
        dataset_size = len(dataset[key_len])

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, dataset_size, batch_size)
        t_samples = [
            np.random.randint(len(dataset["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in dataset.keys():
            transitions[key] = [
                dataset[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            dataset["ob"][episode_idx][t + 1]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]
        transitions["r"] = np.zeros((batch_size,))

        # hindsight experience replay
        for i, (episode_idx, t) in enumerate(zip(episode_idxs, t_samples)):
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_t = np.random.randint(t + 1, len(dataset["ac"][episode_idx]) + 1)
                future_ag = dataset["ag"][episode_idx][future_t]
                if self.reward_func(dataset["ag"][episode_idx][t], future_ag, None) < 0:
                    transitions["g"][i] = future_ag
            transitions["r"][i] = self.reward_func(
                dataset["ag"][episode_idx][t + 1], transitions["g"][i], None
            )

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions
