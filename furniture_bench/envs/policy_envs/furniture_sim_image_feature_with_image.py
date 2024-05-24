import numpy as np
from gym import spaces

import torch

from furniture_bench.config import config
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.envs.legacy_envs.furniture_sim_legacy_env import (
    FurnitureSimEnvLegacy,
)  # Deprecated.
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.robot.robot_state import filter_and_concat_robot_state


class FurnitureSimImageFeatureWithImage(FurnitureSimEnv):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            resize_img=True,
            np_step_out=True,
            channel_first=True,
            gripper_pos_control=True,
            max_env_steps=600,
            **kwargs,
        )

        assert self.num_envs == 1, "FurnitureSimImageFeature supports only 1 env."

        if kwargs["encoder_type"] == "r3m":
            from r3m import load_r3m

            self.layer = load_r3m("resnet50")
            self.embedding_dim = 2048
        elif kwargs["encoder_type"] == "vip":
            from vip import load_vip

            self.layer = load_vip()
            self.embedding_dim = 1024
        self.layer.requires_grad_(False)
        self.layer.eval()

    @property
    def observation_space(self):
        img_shape = (*config["camera"]["resized_img_size"], 3)
        robot_state_dim = 14

        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
            )
        )

    def _get_observation(self):
        obs = super()._get_observation()

        if isinstance(obs["robot_state"], dict):
            # For legacy envs.
            obs["robot_state"] = filter_and_concat_robot_state(obs["robot_state"])

        robot_state = obs["robot_state"].squeeze()
        image1 = obs["color_image1"].squeeze()
        image2 = obs["color_image2"].squeeze()

        # image1 = np.moveaxis(resize(np.moveaxis(image1, 0, -1)), -1, 0)
        # crop_image2 = resize_crop(np.moveaxis(image2, 0, -1))
        # image2 = np.moveaxis(crop_image2, -1, 0)

        with torch.no_grad():
            image1 = torch.tensor(image1).cuda()
            image2 = torch.tensor(image2).cuda()
            image1 = self.layer(image1.unsqueeze(0)).squeeze()
            image2 = self.layer(image2.unsqueeze(0)).squeeze()
            image1 = image1.detach().cpu().numpy()
            image2 = image2.detach().cpu().numpy()
        
        # Make it channel last.
        color_image1 = obs["color_image1"].squeeze()
        color_image2 = obs["color_image2"].squeeze()
        color_image1 = color_image1.transpose(1, 2, 0)
        color_image2 = color_image2.transpose(1, 2, 0)

        return dict(
            robot_state=robot_state,
            color_image1=color_image1,
            color_image2=color_image2,
            image1=image1,
            image2=image2,
        )
