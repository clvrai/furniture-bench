# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

class VIP(nn.Module):
    def __init__(self, device="cuda", lr=1e-4, hidden_dim=1024, size=50, l2weight=1.0, l1weight=1.0, gamma=0.98, num_negatives=0):
        super().__init__()
        self.device = device
        self.l2weight = l2weight
        self.l1weight = l1weight

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.size = size # Resnet size
        self.num_negatives = num_negatives

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
        elif size == 0:
            from transformers import AutoConfig
            self.outdim = 768
            self.convnet = AutoModel.from_config(config = AutoConfig.from_pretrained('google/vit-base-patch32-224-in21k')).to(self.device)

        if self.size == 0:
            self.normlayer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if hidden_dim  > 0:
            self.convnet.fc = nn.Linear(self.outdim, hidden_dim)
        else:
            self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())        

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr = lr)

    ## Forward Call (im --> representation)
    def forward(self, obs, obs_shape = [3, 224, 224]):
        obs_shape = obs.shape[1:]
        # if not already resized and cropped, then add those in preprocessing
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        self.normlayer,
                )
        else:
            preprocess = nn.Sequential(
                        self.normlayer,
                )
        ## Input must be [0, 255], [3,224,224]
        obs = obs.float() /  255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        d = -torch.linalg.norm(tensor1 - tensor2, dim = -1)
        return d
    
