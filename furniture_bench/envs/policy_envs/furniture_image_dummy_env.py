import numpy as np
from gym import spaces

from furniture_bench.config import config
from furniture_bench.envs.policy_envs.furniture_dummy_env import FurnitureDummyEnv


class FurnitureImageDummyEnv(FurnitureDummyEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.robot_state_dim = 14
        self.data_path = f"data/Image/{kwargs['furniture']}.pkl"
        self.use_all_cam = kwargs["use_all_cam"]
        self.stack_cam = kwargs["stack_cam"]
        self.img_shape = (
            9 if self.stack_cam else 3,
            *config["furniture"]["env_img_size"],
        )

    @property
    def observation_space(self):
        if self.use_all_cam and not self.stack_cam:
            return spaces.Dict(
                {
                    "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
                    "image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                    "image2": spaces.Box(-np.inf, np.inf, self.img_shape),
                    "image3": spaces.Box(-np.inf, np.inf, self.img_shape),
                }
            )

        return spaces.Dict(
            {
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
                "image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                "image2": spaces.Box(-np.inf, np.inf, self.img_shape),
            }
        )

    def _get_observation(self):
        if self.use_all_cam and not self.stack_cam:
            return dict(
                robot_state=np.zeros((self.robot_state_dim,)),
                image1=np.zeros(self.img_shape),
                image2=np.zeros(self.img_shape),
                image3=np.zeros(self.img_shape),
            )
        return dict(
            robot_state=np.zeros((self.robot_state_dim,)),
            image1=np.zeros(self.img_shape),
            image2=np.zeros(self.img_shape),
        )
