import numpy as np
import torch
from gym import spaces
from kornia.augmentation import CenterCrop, Resize

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.envs.initialization_mode import load_embedding
from furniture_bench.robot.robot_state import filter_and_concat_robot_state


class FurnitureSimImageWithFeature(FurnitureSimEnv):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            np_step_out=False,
            channel_first=True,
            **kwargs,
        )
        self._resize_img = kwargs["resize_img"]

        device_id = kwargs["compute_device_id"]
        self._device = torch.device(f"cuda:{device_id}")

        self.img_emb_layer, self.embedding_dim = load_embedding(kwargs["encoder_type"], self._device)
        self.reward_img_emb_layer, _ = load_embedding(kwargs["reward_encoder_type"], self._device)

        # Data Augmentation
        if not self._resize_img:
            self.resize = Resize((224, 224))
            img_size = self.img_size
            ratio = 256 / min(img_size[0], img_size[1])
            ratio_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
            self.resize_crop = torch.nn.Sequential(Resize(ratio_size), CenterCrop((224, 224)))

    @property
    def observation_space(self):
        robot_state_dim = 14
        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                color_image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                color_image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
            )
        )

    def _reward(self):
        rewards = super()._reward()
        return rewards.cpu().numpy()

    def _done(self):
        dones = super()._done()
        return dones.cpu().numpy().astype(bool)

    def _get_observation(self):
        obs = super()._get_observation()

        if isinstance(obs["robot_state"], dict):
            # For legacy envs.
            obs["robot_state"] = filter_and_concat_robot_state(obs["robot_state"])

        robot_state = obs["robot_state"]
        image1 = obs["color_image1"]
        image2 = obs["color_image2"]
        if not self._resize_img:
            image1 = self.resize(image1.float())
            image2 = self.resize_crop(image2.float())

        img_emb_image1, img_emb_image2 = (
            self._extract_img_feature(self.img_emb_layer, image1),
            self._extract_img_feature(self.img_emb_layer, image2),
        )
        reward_img_emb_image1, reward_img_emb_image2 = (
            self._extract_img_feature(self.reward_img_emb_layer, image1),
            self._extract_img_feature(self.reward_img_emb_layer, image2),
        )

        return dict(
            robot_state=robot_state.detach().cpu().numpy(),
            image1=img_emb_image1.detach().cpu().numpy(),
            image2=img_emb_image2.detach().cpu().numpy(),
            color_image1=reward_img_emb_image1.detach().cpu().numpy(),
            color_image2=reward_img_emb_image2.detach().cpu().numpy(),
        )

    def _extract_img_feature(self, layer, image):
        with torch.no_grad():
            image_feat = layer(image)
        return image_feat
