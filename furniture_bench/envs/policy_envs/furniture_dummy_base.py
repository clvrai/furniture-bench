import gym

from furniture_bench.config import config


class FurnitureDummyBase(gym.Env):
    """Dummy environment to train the policy."""

    def __init__(self, **kwargs):
        super(FurnitureDummyBase, self).__init__()

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(config["furniture"]["action_dim"],)
        )

    @property
    def observation_space(self):
        raise NotImplementedError

    def step(self, action):
        return self._get_observation(), self._reward(), False, {}

    def _reward(self):
        return 0.0

    def reset(self):
        return self._get_observation()

    def _get_observation(self):
        raise NotImplementedError
