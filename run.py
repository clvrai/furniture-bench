import isaacgym
import torch
import os

import hydra
from omegaconf import OmegaConf, DictConfig

import furniture_bench
from furniture_bench.utils.checkpoint import download_ckpt_if_not_exists


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    if cfg.num_threads > 0:
        print(f"Setting torch.num_threads to {cfg.num_threads}")
        torch.set_num_threads(cfg.num_threads)

    download_ckpt_if_not_exists(cfg.init_ckpt_dir, cfg.run_prefix)

    if cfg['env']['id'] == 'FurnitureSim-v0':
        import isaacgym
    if cfg.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(cfg.gpu)
        cfg.device = cfg.rolf.device = "cuda"
    else:
        cfg.device = cfg.rolf.device = "cpu"

    from rolf.main import Run
    # make config writable
    OmegaConf.set_struct(cfg, False)

    if cfg.rolf.demo_path is not None and not cfg.rolf.demo_path.endswith('/'):
        cfg.rolf.demo_path = cfg.rolf.demo_path + '/'

    cfg.record_video = False
    if cfg.wandb:
        if cfg.wandb_entity is None:
            raise Exception("Specify wandb entity")
        if cfg.wandb_project is None:
            raise Exception("Specify wandb project")
    cfg.rolf.clip_obs = float("inf")  # FIX: str(inf) to float(inf) in hydra
    cfg.rolf.precision = cfg.precision
    cfg.rolf.is_train = cfg.is_train

    # execute training code
    Run(cfg).run()


if __name__ == "__main__":
    main()
