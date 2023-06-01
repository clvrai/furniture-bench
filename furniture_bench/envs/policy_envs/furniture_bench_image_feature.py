import numpy as np
from gym import spaces

import torch

from furniture_bench.envs.furniture_bench_env import FurnitureBenchEnv
from furniture_bench.config import config
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from furniture_bench.robot.panda import PandaError


class FurnitureBenchImageFeature(FurnitureBenchEnv):
    def __init__(self, **kwargs):
        super().__init__(
            furniture=kwargs["furniture"],
            from_skill=kwargs["from_skill"],
            high_random_idx=kwargs["high_random_idx"],
            randomness=kwargs["randomness"],
        )

        self.img_shape = (*config["furniture"]["env_img_size"], 3)
        self.robot_state_dim = 14

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
        return spaces.Dict(
            {
                "robot_state": spaces.Box(-np.inf, np.inf, (self.robot_state_dim,)),
                "image1": spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                "image2": spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
            }
        )

    def _get_observation(self):
        """If successful, returns (obs, True); otherwise, returns (None, False)."""
        obs, panda_error = super()._get_observation()
        robot_state = obs["robot_state"]
        image1 = np.moveaxis(obs["color_image1"], -1, 0).astype(np.float32)
        image2 = np.moveaxis(obs["color_image2"], -1, 0).astype(np.float32)

        with torch.no_grad():
            image1 = self.layer(torch.tensor(image1, device="cuda").unsqueeze(0))
            image1 = image1.squeeze().detach().cpu().numpy()
            image2 = self.layer(torch.tensor(image2, device="cuda").unsqueeze(0))
            image2 = image2.squeeze().detach().cpu().numpy()

        return (
            dict(
                robot_state=filter_and_concat_robot_state(robot_state),
                image1=image1,
                image2=image2,
            ),
            panda_error,
        )
