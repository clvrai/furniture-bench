import gym
from gym import spaces

from furniture_bench.config import config


class FurnitureDummyEnv(gym.Env):
    """Dummy environment to train the policy."""

    def __init__(self, **kwargs):
        super(FurnitureDummyEnv, self).__init__()

    @property
    def action_space(self):
        return spaces.Box(
            low=-1.0, high=1.0, shape=(config["furniture"]["action_dim"],)
        )

    @property
    def observation_space(self):
        raise NotImplementedError

    def step(self, action):
        return self._get_obs(), self._reward(), False, {}

    def _reward(self):
        return 0.0

    def reset(self):
        return self._get_obs()

    def _get_observation(self):
        raise NotImplementedError
