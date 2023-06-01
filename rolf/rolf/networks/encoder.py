"""
Code reference:
  https://github.com/MishaLaskin/rad/blob/master/encoder.py
"""

import gym.spaces
from rolf.utils.aug import RandomShiftsAug
import torch
import torch.nn as nn

from .utils import CNN, R3M, VIP, ResNet18


class Encoder(nn.Module):

    def __init__(self, cfg, ob_space):
        super().__init__()

        self.cfg = cfg
        self._encoder_type = cfg.encoder_type
        self._ob_space = ob_space
        self.aug = RandomShiftsAug()

        self.base = nn.ModuleDict()
        encoder_output_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) in [3, 4]:
                if self._encoder_type == "mlp":
                    self.base[k] = None
                    encoder_output_dim += gym.spaces.flatdim(v)
                elif self._encoder_type == 'r3m':
                    r3m = R3M(cfg)
                    self.base[k] = r3m
                    encoder_output_dim += 2048
                elif self._encoder_type == 'vip':
                    vip = VIP(cfg)
                    self.base[k] = vip
                    encoder_output_dim += 1024
                elif self._encoder_type == 'resnet18':
                    resnet18 = ResNet18(cfg)
                    self.base[k] = resnet18
                    encoder_output_dim += 512
                else:
                    if len(v.shape) == 3:
                        image_dim = v.shape[0]
                    elif len(v.shape) == 4:
                        image_dim = v.shape[0] * v.shape[1]
                    self.base[k] = CNN(cfg, image_dim)
                    encoder_output_dim += self.base[k].output_dim
            elif len(v.shape) == 1:
                self.base[k] = None
                encoder_output_dim += gym.spaces.flatdim(v)
            else:
                raise ValueError("Check the shape of observation %s (%s)" % (k, v))

        self.output_dim = encoder_output_dim

    def forward(self, ob, detach_conv=False):
        encoder_outputs = []
        for k, v in ob.items():
            if self.base[k] is not None:
                if isinstance(self.base[k], CNN):
                    if v.max() > 1.0:
                        v = v.float() / 255.0
                if isinstance(self.base[k], ResNet18):
                    # preprocess = transforms.Compose([
                    #     transforms.ToTensor(),
                    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # ])
                    # import pdb
                    # pdb.set_trace()
                    # Preprocess
                    if len(v.shape) == 5:
                        v.squeeze_(1)
                    v = v / 255.0
                    v[:, 0, :, :] = (v[:, 0, :, :] - 0.485) / 0.229
                    v[:, 1, :, :] = (v[:, 1, :, :] - 0.456) / 0.224
                    v[:, 2, :, :] = (v[:, 2, :, :] - 0.406) / 0.225
                if len(v.shape) == 4:
                    # Image
                    v = v.float()
                    if self.cfg.image_agmt:
                        v = self.aug(v)
                encoder_outputs.append(self.base[k](v, detach_conv=detach_conv))
            else:
                encoder_outputs.append(v.flatten(start_dim=1))
        out = torch.cat(encoder_outputs, dim=-1)
        assert len(out.shape) == 2
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for k in self.base.keys():
            if self.base[k] is not None:
                self.base[k].copy_conv_weights_from(source.base[k])
