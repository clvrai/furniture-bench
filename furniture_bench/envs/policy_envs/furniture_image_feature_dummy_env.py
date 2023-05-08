"""Environment that gives pre-trained image features as observation.""" ""
import numpy as np
from gym import spaces

from furniture_bench.config import config
from furniture_bench.envs.policy_envs.furniture_dummy_env import FurnitureDummyEnv


class FurnitureImageFeatureDummyEnv(FurnitureDummyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.img_shape = (*config["furniture"]["env_img_size"], 3)
        if not kwargs["use_encoder"]:
            self.embedding_dim = (
                2048 if kwargs["encoder_type"] == "r3m" else 1024
            )  # The dimension of R3M ResNet50 embedding or VIP.
        self.robot_state_dim = 8 + 6
        self.use_encoder = kwargs["use_encoder"]

    @property
    def observation_space(self):
        if self.use_encoder:
            return spaces.Dict(
                {
                    "image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                    "image2": spaces.Box(-np.inf, np.inf, self.img_shape),
                    "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
                }
            )

        return spaces.Dict(
            {
                "image1": spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                "image2": spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
            }
        )
