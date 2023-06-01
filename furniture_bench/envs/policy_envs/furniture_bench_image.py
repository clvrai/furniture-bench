import numpy as np
from gym import spaces

from furniture_bench.envs.furniture_bench_env import FurnitureBenchEnv
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.robot.robot_state import (
    filter_and_concat_robot_state,
    FILTERED_ROBOT_STATE_DIM,
)
from furniture_bench.robot.panda import PandaError


class FurnitureBenchImage(FurnitureBenchEnv):
    """Furniture environment with image observation."""

    def __init__(self, **kwargs):
        super().__init__(
            furniture=kwargs["furniture"],
            randomness=kwargs["randomness"],
            from_skill=kwargs["from_skill"],
            to_skill=kwargs["to_skill"],
            high_random_idx=kwargs["high_random_idx"],
        )

        self.img_shape = (3, *config["furniture"]["env_img_size"])
        self.use_all_cam = kwargs["use_all_cam"]
        self.img_shape = (*config["furniture"]["env_img_size"], 3)

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "robot_state": spaces.Box(-np.inf, np.inf, (FILTERED_ROBOT_STATE_DIM,)),
                "image1": spaces.Box(-np.inf, np.inf, self.img_shape),
                "image2": spaces.Box(-np.inf, np.inf, self.img_shape),
            }
        )

    def _get_observation(self):
        """If successful, returns (obs, True); otherwise, returns (None, False)."""
        robot_state, panda_error = self.robot.get_state()
        _, _, image1, _, image2, _, _, _ = self.furniture.get_parts_poses()

        # change from (H, W, C) to (C, H, W)
        image1 = np.moveaxis(resize(image1), -1, 0)
        image2 = np.moveaxis(resize_crop(image2), -1, 0)

        return (
            dict(
                robot_state=filter_and_concat_robot_state(robot_state.__dict__),
                image1=image1,
                image2=image2,
            ),
            panda_error,
        )
