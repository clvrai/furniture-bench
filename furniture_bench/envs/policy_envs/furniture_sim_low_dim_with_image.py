import numpy as np
from gym import spaces

from furniture_bench.config import config
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.envs.legacy_envs.furniture_sim_legacy_env import (
    FurnitureSimEnvLegacy,
)  # Deprecated.
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from furniture_bench.envs.observation import DEFAULT_VISUAL_STATE_OBS


class FurnitureSimLowDimWithImage(FurnitureSimEnv):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            resize_img=True,
            np_step_out=True,
            channel_first=True,
            gripper_pos_control=True,
            max_env_steps=600,
            obs_keys=DEFAULT_VISUAL_STATE_OBS,
            **kwargs,
        )

        assert self.num_envs == 1, "FurnitureSimImageFeature supports only 1 env."
        self.gpu = kwargs["gpu"]

    @property
    def observation_space(self):
        robot_state_dim = 14

        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                parts_poses=spaces.Box(-np.inf, np.inf, (5 * 7,)),
            )
        )

    def _get_observation(self):
        obs = super()._get_observation()

        if isinstance(obs["robot_state"], dict):
            # For legacy envs.
            obs["robot_state"] = filter_and_concat_robot_state(obs["robot_state"])

        # Make it channel last.
        color_image1 = obs["color_image1"].squeeze()
        color_image2 = obs["color_image2"].squeeze()
        color_image1 = color_image1.transpose(1, 2, 0)
        color_image2 = color_image2.transpose(1, 2, 0)

        robot_state = obs["robot_state"].squeeze()
        parts_poses = obs["parts_poses"].squeeze()
        return dict(
            robot_state=robot_state,
            parts_poses=parts_poses,
            color_image1=color_image1,
            color_image2=color_image2,
        )
