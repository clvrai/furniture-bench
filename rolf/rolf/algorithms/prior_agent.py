import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .base_agent import BaseAgent
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl, ClSPiRLMdl
from data.dataloader import (
    GlobalSplitVideoDataset,
)  # need to set the path when running dreamer
from spirl.utils.general_utils import AttrDict, map_dict
from ..utils import Logger, Info
from ..utils.mpi import mpi_average
from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    sync_network,
    sync_grad,
    to_tensor,
)


class PriorAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.global_step = 0
        self.pbar = None

        if cfg.img:
            self._actor = ImageClSPiRLMdl(cfg)
        else:
            self._actor = ClSPiRLMdl(cfg)
        self._network_cuda()
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=cfg.lr)
        self._actor_lr_scheduler = StepLR(
            self._actor_optim, step_size=cfg.max_global_step // 5, gamma=0.5
        )

        if cfg.is_train:
            self._train_loader = self._get_dataset(cfg, "train", cfg.data.n_repeat, -1)
        self._val_loader = self._get_dataset(cfg, "val", 1, cfg.data.val_dataset_size)

        self._log_creation()

    def _get_dataset(self, cfg, phase, n_repeat, dataset_size=-1):
        data_dir = os.path.join("data", cfg.data.dir)

        loader = GlobalSplitVideoDataset(
            data_dir,
            cfg,
            resolution=self._actor.resolution,
            phase=phase,
            shuffle=phase == "train",
            dataset_size=dataset_size,
        ).get_data_loader(cfg.batch_size, n_repeat)

        return loader

    def _log_creation(self):
        Logger.info("Creating a Prior agent")
        Logger.info("The actor has %d parameters", count_parameters(self._actor))

    def state_dict(self):
        return {
            "actor_state_dict": self._actor.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._network_cuda()

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._device)

    def _network_cuda(self):
        self._actor.to(self._device)

    def is_off_policy(self):
        return False

    def update(self):
        train_info = Info()
        for sample_batched in self._train_loader:
            _train_info = self._update_network(sample_batched, train=True)
            train_info.add(_train_info)
            self.pbar.update(1)
        self._actor_lr_scheduler.step()

        train_info = train_info.get_dict(only_scalar=True)
        Logger.info("Prior loss %f", train_info["actor_loss"])
        return train_info

    def evaluate(self):
        if self._val_loader:
            eval_info = Info()
            for transitions in self._val_loader:
                _eval_info = self._update_network(transitions, train=False)
                eval_info.add(_eval_info)
            return eval_info.get_dict(only_scalar=True)
        Logger.warning("No validation set available, make sure '--val_split' is set")
        return None

    def _update_network(self, sample_batched, train=True):
        info = Info()
        inputs = AttrDict(map_dict(lambda x: x.to(self._device), sample_batched))

        # the actor loss
        output = self._actor(inputs)
        losses = self._actor.loss(output, inputs)

        info["actor_loss"] = losses.total.value.cpu().item()
        info["rec_mse_loss"] = losses.rec_mse.value.cpu().item()
        info["kl_loss"] = losses.kl_loss.value.cpu().item()
        info["q_hat_loss"] = losses.q_hat_loss.value.cpu().item()
        info["beta"] = self._actor.beta
        info["pred_ac"] = output.reconstruction.cpu().detach()
        info["GT_ac"] = inputs.actions.cpu()

        if train:
            # update the actor
            self._actor_optim.zero_grad()
            losses.total.value.backward()
            sync_grad(self._actor, self._device)
            if self.global_step < self._cfg.init_grad_clip_step:
                # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(
                    self._actor.parameters(), self._cfg.init_grad_clip
                )
            self._actor_optim.step()

        self.global_step += 1

        return mpi_average(info.get_dict(only_scalar=True))
