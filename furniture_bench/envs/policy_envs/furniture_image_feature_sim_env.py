import numpy as np
from gym import spaces

import torch
from r3m import load_r3m
from vip import load_vip

from furniture_bench.config import config
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.robot.misc import concat_robot_state


class FurnitureImageFeatureSimEnv(FurnitureSimEnv):
    def __init__(self, **kwargs):
        super().__init__(
            furniture=kwargs["furniture"],
            resize_img=False,
            np_step_out=True,
            record=kwargs["record"],
            channel_first=True,
        )

        self.img_shape = (*config["furniture"]["env_img_size"], 3)
        self.robot_state_dim = 14
        self.data_path = f"data/Image/{kwargs['furniture']}.pkl"

        self.use_encoder = kwargs["use_encoder"]
        if not kwargs["use_encoder"]:
            if kwargs["encoder_type"] == "r3m":
                self.embedding_dim = 2048
                self.layer = load_r3m("resnet50")
            elif kwargs["encoder_type"] == "vip":
                self.layer = load_vip()
                self.embedding_dim = 1024
            self.layer.requires_grad_(False)
            self.layer.eval()

        self.robot_state_dim = 8 + 6

        self.noise_counter = -1
        self.max_noise_counter = 5

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

    def _get_observation(self):
        """If successful, return (obs, True). Otherwise return (None, False)."""
        obs = super()._get_observation()
        image1 = obs["color_image1"].squeeze()
        image2 = obs["color_image2"].squeeze()

        image1 = torch.tensor(np.moveaxis(resize(np.moveaxis(image1, 0, -1)), -1, 0))
        image2 = torch.tensor(
            np.moveaxis(
                resize_crop(
                    np.moveaxis(image2, 0, -1), config["camera"]["color_img_size"]
                ),
                -1,
                0,
            ),
        )

        robot_state = obs["robot_state"]
        for k, v in robot_state.items():
            robot_state[k] = v.squeeze()

        if not self.use_encoder:
            with torch.no_grad():
                image1 = image1.cuda()
                image2 = image2.cuda()
                image1 = (
                    self.layer(image1.unsqueeze(0)).squeeze().detach().cpu().numpy()
                )
                image2 = (
                    self.layer(image2.unsqueeze(0)).squeeze().detach().cpu().numpy()
                )

            return dict(
                image1=image1,
                image2=image2,
                robot_state=concat_robot_state(robot_state),
            )

        return dict(
            image1=image1.detach().cpu().numpy(),
            image2=image2.deteach().cpu().numpy(),
            robot_state=concat_robot_state(robot_state),
        )
