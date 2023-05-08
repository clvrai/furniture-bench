import numpy as np
from gym import spaces

from furniture_bench.envs.furniture_env import FurnitureEnv
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.robot.misc import concat_robot_state
from furniture_bench.robot.panda import PandaError
from furniture_bench.envs.initialization_mode import str_to_enum


class FurnitureImageEnv(FurnitureEnv):
    """Furniture environment with image observation."""

    def __init__(self, **kwargs):
        super().__init__(
            furniture=kwargs["furniture"],
            randomness=str_to_enum(kwargs["randomness"]),
            from_skill=kwargs["from_skill"],
            skill=kwargs["skill"],
            high_random_idx=kwargs["high_random_idx"],
        )

        self.img_shape = (3, *config["furniture"]["env_img_size"])
        self.robot_state_dim = 14
        self.use_all_cam = kwargs["use_all_cam"]
        self.stack_cam = False
        self.img_shape = (
            9 if self.stack_cam else (*config["furniture"]["env_img_size"], 3)
        )

        self.noise_counter = -1
        self.max_noise_counter = 3

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
                "image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                "image2": spaces.Box(-np.inf, np.inf, self.img_shape),
            }
        )

    def _get_observation(self):
        """If successful, return (obs, True). Otherwise return (None, False)."""
        robot_state, _ = self.robot.get_state()
        _, _, image1, _, image2, _, _, _ = self.furniture.get_parts_poses()

        image1 = np.moveaxis(resize(image1), -1, 0) # change from (H, W, C) to (C, H, W)
        image2 = np.moveaxis(
            resize_crop(image2, config["camera"]["color_img_size"]), -1, 0
        )

        return (
            dict(
                image1=image1,
                image2=image2,
                robot_state=concat_robot_state(robot_state.__dict__),
            ),
            PandaError.OK,
        )
