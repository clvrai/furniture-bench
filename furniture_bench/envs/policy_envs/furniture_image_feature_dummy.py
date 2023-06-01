import numpy as np
from gym import spaces

from furniture_bench.config import config
from furniture_bench.envs.policy_envs.furniture_dummy_base import FurnitureDummyBase


class FurnitureImageFeatureDummy(FurnitureDummyBase):
    """Dummy environment with pre-trained image features as observation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.robot_state_dim = 8 + 6
        self.img_shape = (*config["furniture"]["env_img_size"], 3)
        self.embedding_dim = (
            2048 if kwargs["encoder_type"] == "r3m" else 1024
        )  # The dimension of R3M ResNet50 embedding or VIP.

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "image1": spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                "image2": spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
            }
        )
