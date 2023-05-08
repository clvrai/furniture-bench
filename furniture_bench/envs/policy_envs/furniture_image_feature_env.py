import numpy as np
from gym import spaces

import torch
from r3m import load_r3m
from vip import load_vip

from furniture_bench.envs.furniture_env import FurnitureEnv
from furniture_bench.config import config
from furniture_bench.robot.misc import concat_robot_state
from furniture_bench.robot.panda import PandaError
from furniture_bench.envs.initialization_mode import str_to_enum


class FurnitureImageFeatureEnv(FurnitureEnv):
    def __init__(self, **kwargs):
        super().__init__(
            furniture=kwargs["furniture"],
            from_skill=kwargs["from_skill"],
            high_random_idx=kwargs["high_random_idx"],
            randomness=str_to_enum(kwargs["randomness"]),
        )

        self.img_shape = (*config["furniture"]["env_img_size"], 3)
        self.robot_state_dim = 14
        self.data_path = f"data/Image/{kwargs['furniture']}.pkl"

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

        self.use_encoder = kwargs["use_encoder"]

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
        obs, _ = super()._get_observation()
        robot_state = obs["robot_state"]
        image1 = np.moveaxis(obs["color_image1"], -1, 0).astype(np.float32)
        image2 = np.moveaxis(obs["color_image2"], -1, 0).astype(np.float32)

        if not self.use_encoder:
            with torch.no_grad():
                image1 = (
                    self.layer(torch.tensor(image1, device="cuda").unsqueeze(0))
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                image2 = (
                    self.layer(torch.tensor(image2, device="cuda").unsqueeze(0))
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

            return (
                dict(
                    image1=image1,
                    image2=image2,
                    robot_state=concat_robot_state(robot_state),
                ),
                PandaError.OK,
            )
            # return np.concatenate([embedding, concat_robot_state(robot_state.__dict__)]), PandaError.OK
        return (
            dict(
                image1=image1,
                image2=image2,
                robot_state=concat_robot_state(robot_state),
            ),
            PandaError.OK,
        )
