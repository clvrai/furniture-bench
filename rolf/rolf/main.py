""" Launch RL/IL training and evaluation. """

import sys
import signal
import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import hydra
from six.moves import shlex_quote
from mpi4py import MPI
from omegaconf import OmegaConf, DictConfig

from .trainer import Trainer
from .utils import Logger
from .utils.mpi import mpi_sync


class Run(object):
    """Class for experiment run."""

    def __init__(self, cfg):
        self._cfg = cfg

        self._set_run_name()
        self._set_log_path()
        self._set_mpi_workers()

        # set global seed
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        os.environ["DISPLAY"] = ":1"

    def run(self):
        """Runs Trainer."""
        trainer = self._get_trainer()
        if self._cfg.is_train:
            Logger.info("Start training")
            trainer.train()
            Logger.info("Finish training")
        else:
            Logger.info("Start evaluating")
            trainer.evaluate()
            Logger.info("Finish evaluating")

    def _set_run_name(self):
        """Sets run name."""
        cfg = self._cfg
        cfg.run_name = f"{cfg.env.id}.{cfg.rolf.name}.{cfg.run_prefix}.{cfg.seed}"

    def _set_log_path(self):
        """Sets paths to log directories."""
        cfg = self._cfg
        log_dir = Path(cfg.log_root_dir).expanduser() / cfg.run_name
        cfg.log_dir = str(log_dir)
        cfg.video_dir = str(log_dir / "video")
        cfg.demo_dir = str(log_dir / "demo")
        cfg.ckpt_dir = str(log_dir / "ckpt")
        cfg.replay_dir = str(log_dir / "replay")

    def _make_log_files(self):
        """Sets up log directories and saves git diff and command line."""
        cfg = self._cfg

        Logger.info(f"Create log directory: {cfg.log_dir}")
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.video_dir).mkdir(exist_ok=True)
        Path(cfg.demo_dir).mkdir(exist_ok=True)
        Path(cfg.ckpt_dir).mkdir(exist_ok=True)
        Path(cfg.replay_dir).mkdir(exist_ok=True)

        if cfg.is_train:
            timestamp = time.strftime("%Y-%d-%m_%H-%M-%S")

            # log git diff
            log_dir = Path(cfg.log_dir)
            git_path = log_dir / f"git_{timestamp}.txt"
            cmd_path = log_dir / f"cmd_{timestamp}.sh"
            args = " ".join([shlex_quote(arg) for arg in sys.argv[1:]])
            cmds = [
                f"echo `git rev-parse HEAD` >> {git_path}",
                f"git diff --submodule=diff >> {git_path}",
                f"echo 'python -m rolf.main {args}' >> {cmd_path}",
            ]
            os.system("\n".join(cmds))

            # log config
            param_path = log_dir / f"params_{timestamp}.yaml"
            Logger.info(f"Store parameters in {param_path}")
            param_path.write_text(OmegaConf.to_yaml(cfg))

            # setup wandb
            exclude = ["device"]

            def flat_cfg(x, prefix=""):
                ret = {}
                for k, v in x.items():
                    if isinstance(v, dict):
                        ret.update(flat_cfg(v, f"{prefix}{k}/"))
                    elif k not in exclude:
                        ret[f"{prefix}{k}"] = v
                return ret

            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                resume=cfg.run_name,
                config=flat_cfg(cfg),
                dir=cfg.log_dir,
                notes=cfg.notes,
                mode="online" if cfg.wandb else "disabled",
            )
            wandb.save(str(git_path))
            # dumps to yaml string
            yaml_data: str = OmegaConf.to_yaml(cfg)
            # dumps to file:
            name = os.path.join('config', f"{cfg.run_name}.yaml")
            if not os.path.isfile(name):
                with open(name, "w") as f:
                    OmegaConf.save(cfg, f)
                wandb.save(name)

    def _set_mpi_workers(self):
        cfg = self._cfg

        rank = MPI.COMM_WORLD.Get_rank()
        cfg.rank = rank
        cfg.is_chef = rank == 0
        cfg.num_workers = MPI.COMM_WORLD.Get_size()

        cfg.seed += rank
        if hasattr(cfg.env, "unity"):
            cfg.env.unity.port += rank * 2  # training env + evaluation env

        if cfg.is_chef:
            Logger.warning("Run a base worker.")
            self._make_log_files()
        else:
            Logger.warning(f"Run worker {cfg.rank} and disable Logger.")
            Logger.setLevel(logging.CRITICAL)

        # syncronize all processes
        mpi_sync()

        def shutdown(signal, frame):
            Logger.warning(f"Received signal {signal}: exiting")
            sys.exit(128 + signal)

        signal.signal(signal.SIGHUP, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

    def _get_trainer(self):
        return Trainer(self._cfg)


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # make config writable
    OmegaConf.set_struct(cfg, False)

    # change default config
    cfg.wandb_entity = "clvr"
    cfg.wandb_project = "p-skill-model"  # change it for now to record dreamer runs
    cfg.rolf.precision = cfg.precision

    # execute training code
    Run(cfg).run()


if __name__ == "__main__":
    main()
