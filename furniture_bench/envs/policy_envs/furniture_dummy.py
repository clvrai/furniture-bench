import numpy as np
from gym import spaces

from furniture_bench.config import config
from furniture_bench.envs.policy_envs.furniture_dummy_base import FurnitureDummyBase


class FurnitureDummy(FurnitureDummyBase):
    """Furniture environment with image observation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.robot_state_dim = 14
        self.img_shape = (3, *config["furniture"]["env_img_size"])

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
                "color_image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                "color_image2": spaces.Box(-np.inf, np.inf, self.img_shape),
            }
        )

    def _get_observation(self):
        return dict(
            robot_state=np.zeros((self.robot_state_dim,)),
            color_image1=np.zeros(self.img_shape),
            color_image2=np.zeros(self.img_shape),
        )
