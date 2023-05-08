import numpy as np
from gym import spaces

from furniture_bench.config import config
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.perception.image_utils import resize_crop, resize
from furniture_bench.robot.misc import concat_robot_state


class FurnitureImageSimEnv(FurnitureSimEnv):
    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(
            furniture=kwargs["furniture"],
            record=True,
            resize_img=False,
            np_step_out=True,
        )

        self.img_shape = (3, *config["furniture"]["env_img_size"])
        self.robot_state_dim = 14

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                "image2": spaces.Box(-np.inf, np.inf, self.img_shape),
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
            }
        )

    def _get_observation(self):
        """If successful, return (obs, True). Otherwise return (None, False)."""

        obs = super()._get_observation()
        image1 = obs["color_image1"].squeeze()
        image2 = obs["color_image2"].squeeze()

        image1 = np.moveaxis(resize(np.moveaxis(image1, 0, -1)), -1, 0)
        image2 = np.moveaxis(
            resize_crop(np.moveaxis(image2, 0, -1), config["camera"]["color_img_size"]),
            -1,
            0,
        )

        robot_state = obs["robot_state"]
        for k, v in robot_state.items():
            robot_state[k] = v.squeeze()

        return dict(
            image1=image1, image2=image2, robot_state=concat_robot_state(robot_state)
        )
