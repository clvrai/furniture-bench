import pickle
from collections import deque, OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import gym.spaces

from ..utils import Logger
from ..utils.gym_env import get_non_absorbing_state, get_absorbing_state, zero_value


class ExpertDataset(Dataset):
    """Dataset class for Imitation Learning."""

    def __init__(
        self,
        path,
        subsample_interval=1,
        ac_space=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        use_low_level=False,
        sample_range=(0.0, 1.0),
        num_demos=None,
        keys=None,
    ):
        self.train = train  # training set or test set
        self.initial_states = deque(maxlen=10000)
        self.initial_obs = deque(maxlen=10000)
        self.terminal_obs = deque(maxlen=10000)
        self.num_demos = num_demos

        self._data = []
        self._ac_space = ac_space
        self._keys = keys or ["ob", "ac", "done"]
        is_ob_next = "ob_next" in self._keys

        assert (
            path is not None
        ), "--demo_path should be set (e.g. demos/Sawyer_toy_table)"
        demo_files = self._get_demo_files(path)
        num_demos = 0

        # Load the pickled numpy arrays.
        for file_path in demo_files:
            if num_demos == self.num_demos:
                break

            with open(file_path, "rb") as f:
                demos = pickle.load(f)
            if not isinstance(demos, list):
                demos = [demos]

            for demo in demos:
                if num_demos == self.num_demos:
                    break

                if is_ob_next and len(demo["observation"]) != len(demo["actions"]) + 1:
                    Logger.error(
                        f"Mismatch in # of observations ({len(demo['observations'])}) and actions ({len(demo['actions'])}): {str(file_path)}"
                    )
                    continue

                offset = np.random.randint(0, subsample_interval)
                num_demos += 1

                if use_low_level:
                    length = len(demo["low_level_actions"])
                    start = int(length * sample_range[0])
                    end = int(length * sample_range[1])
                    self.initial_states.append(demo["states"][0])
                    self.initial_obs.append(demo["low_level_obs"][0])
                    self.terminal_obs.append(demo["low_level_obs"][-1])
                    for i in range(start + offset, end, subsample_interval):
                        transition = {"ob": demo["low_level_obs"][i]}
                        if "ob_next" in self._keys:
                            transition["ob_next"] = demo["low_level_obs"][i + 1]
                        if isinstance(demo["low_level_actions"][i], dict):
                            transition["ac"] = demo["low_level_actions"][i]
                        else:
                            transition["ac"] = gym.spaces.unflatten(
                                ac_space, demo["low_level_actions"][i]
                            )
                        transition["done"] = 1 if i + 1 == length else 0
                        self._data.append(transition)
                    continue

                length = len(demo["actions"])
                start = int(length * sample_range[0])
                end = int(length * sample_range[1])
                for i in range(start + offset, end, subsample_interval):
                    transition = {"ob": demo["observations"][i]}
                    if isinstance(demo["actions"][i], dict):
                        transition["ac"] = demo["actions"][i]
                    else:
                        transition["ac"] = gym.spaces.unflatten(
                            ac_space, demo["actions"][i]
                        )
                    if "dones" in demo:
                        transition["done"] = int(demo["dones"][i])
                    else:
                        transition["done"] = 1 if i + 1 == length else 0
                    if "ob_next" in self._keys:
                        transition["ob_next"] = demo["observations"][i + 1]
                    if "rew" in self._keys and "rewards" in demo:
                        transition["rew"] = demo["rewards"][i]
                    self._data.append(transition)

        Logger.warning(
            f"Load {num_demos} demonstrations with {len(self._data)} states from {len(demo_files)} files",
        )

        for i in range(len(self._data)):
            for k, v in self._data[i]["ob"].items():
                if v.dtype == np.float64:
                    self._data[i]["ob"][k] = v.astype(np.float32)

            self._data[i]["ob"] = {
                k: torch.tensor(v) for k, v in self._data[i]["ob"].items()
            }
            self._data[i]["ac"] = OrderedDict(
                [(k, torch.tensor(v)) for k, v in self._data[i]["ac"].items()]
            )

    def add_absorbing_states(self, ob_space, ac_space):
        new_data = []
        absorbing_state = get_absorbing_state(ob_space)
        absorbing_action = zero_value(ac_space, dtype=np.float32)
        for i in range(len(self._data)):
            transition = self._data[i].copy()
            transition["ob"] = get_non_absorbing_state(self._data[i]["ob"])
            # learn reward for the last transition regardless of timeout (different from paper)
            if self._data[i]["done"]:
                transition["ob_next"] = absorbing_state
                transition["done_mask"] = 0  # -1 absorbing, 0 done, 1 not done
            else:
                transition["ob_next"] = get_non_absorbing_state(
                    self._data[i]["ob_next"]
                )
                transition["done_mask"] = 1  # -1 absorbing, 0 done, 1 not done
            new_data.append(transition)

            if self._data[i]["done"]:
                transition = {
                    "ob": absorbing_state,
                    "ob_next": absorbing_state,
                    "ac": absorbing_action,
                    # "rew": np.float64(0.0),
                    "done": 0,
                    "done_mask": -1,  # -1 absorbing, 0 done, 1 not done
                }
                new_data.append(transition)

        self._data = new_data

    def _get_demo_files(self, demo_file_path):
        demos = []
        p = Path(".")
        if not demo_file_path.endswith(".pkl"):
            demo_file_path = demo_file_path + "*.pkl"
        for f in p.glob(demo_file_path):
            demos.append(str(f))
        return demos

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (ob, ac) where target is index of the target class.
        """
        return self._data[index]

    def __len__(self):
        return len(self._data)
