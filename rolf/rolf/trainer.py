"""
Base code for RL/IL training.
Collects rollouts and updates policy networks.
"""

import pickle
from pathlib import Path

import torch
import wandb
import h5py
import imageio
import numpy as np
from tqdm import tqdm

from .algorithms import RL_ALGOS, IL_ALGOS
from .utils import Logger, Every, StopWatch, Info, LOG_TYPES, make_env
from .utils.pytorch import get_ckpt_path, check_memory_kill_switch
from .utils.mpi import mpi_sum, mpi_gather_average


class Trainer(object):
    """
    Trainer class for PPO, SAC, DDPG, TD3, BC, GAIL, and Dreamer in PyTorch.
    """

    def __init__(self, cfg):
        """Initializes class with the configuration."""
        self._cfg = cfg
        self._is_chef = cfg.is_chef

        # Create environment for training.
        self._env = make_env(cfg.env.id, cfg.env, cfg.seed, cfg.rolf.name)
        ob_space = self._env.observation_space
        ac_space = self._env.action_space
        Logger.info(f"Observation space: {ob_space}")
        Logger.info(f"Action space: {ac_space}")

        # Create environment for evaluation.
        if cfg.rolf.name in ["bc", "iris"]:
            self._env_eval = self._env  # No training env for BC.
        else:
            cfg_eval = cfg.copy()
            if hasattr(cfg_eval.env, "unity"):
                cfg_eval.env.unity.port += 1
            self._env_eval = make_env(cfg.env.id, cfg_eval.env, cfg.seed, cfg.rolf.name)

        # Build agent and networks for algorithm.
        self._agent = self._get_agent_by_name(cfg.rolf.name)(
            cfg.rolf, ob_space, ac_space
        )

        # Build rollout runner.
        self._runner = self._agent.get_runner(cfg, self._env, self._env_eval)

    def train(self):
        """Trains an agent."""
        cfg = self._cfg

        print(self._agent)
        # Load checkpoint.
        ckpt_info = self._load_ckpt(cfg.init_ckpt_path, cfg.ckpt_num)
        step = ckpt_info.get("step", 0)

        # Sync the networks across the cpus.
        self._agent.sync_networks()

        # Decide how many episodes or how long rollout to collect.
        runner = self._runner.run(every_steps=cfg.rolf.train_every, step=step)

        Logger.info(f"Start training at step={step}")
        if self._is_chef:
            pbar = tqdm(initial=step, total=cfg.rolf.max_global_step, desc=cfg.run_name)
            ep_info = Info()
            train_info = Info()
            should_log = Every(cfg.rolf.log_every, step)
            should_evaluate = Every(cfg.rolf.evaluate_every, step)
            should_ckpt = Every(cfg.rolf.ckpt_every, step)
            timer = StopWatch(step)

        # Collect warm-up rollouts.
        if step < cfg.rolf.warm_up_steps:
            self._agent.warm_up_training.reset()

        while step < cfg.rolf.warm_up_steps:
            rollout, rollout_steps, info = next(runner)
            self._agent.store_episode(rollout)
            rollout_steps = mpi_sum(rollout_steps)
            step += rollout_steps
            if step < cfg.rolf.max_ob_norm_step:
                self._update_normalizer(rollout)
            if self._is_chef:
                pbar.update(rollout_steps)

        if cfg.rolf.name == "bc" and cfg.rolf.ob_norm:
            self._agent.update_normalizer()

        while step < cfg.rolf.max_global_step:
            # Collect training rollout (do nothing for BC).
            rollout, rollout_steps, info = next(runner)
            info = mpi_gather_average(info)
            rollout_steps = mpi_sum(rollout_steps)
            if rollout:
                self._agent.store_episode(rollout)

            # Train agent.
            _train_info = self._agent.update()

            if runner and step < cfg.rolf.max_ob_norm_step:
                self._update_normalizer(rollout)

            step += rollout_steps
            self._agent.set_step(step)

            # Log training and episode information.
            if not self._is_chef:
                continue

            pbar.update(rollout_steps)
            ep_info.add(info)
            train_info.add(_train_info)

            if should_log(step):
                train_info.add({"steps_per_sec": timer(step)})
                self._log_train(step, train_info.get_dict(), ep_info.get_dict())

            if should_evaluate(step):
                Logger.info(f"Evaluate at step={step}")
                if cfg.rolf.name in ["bc", "iris"]:
                    # Validation set.
                    ep_info = self._agent.evaluate()
                    # self._log_test(step, ep_info)
                else:
                    _, ep_info = self._evaluate(step, cfg.record_video)
                self._log_test(step, ep_info.get_dict())

            if should_ckpt(step):
                self._save_ckpt(step)

        # Store the final model.
        if self._is_chef:
            self._save_ckpt(step)

        Logger.info(f"Reached {step} steps. Worker {cfg.rank} stopped.")

    def evaluate(self):
        """Evaluates an agent stored in chekpoint with `cfg.ckpt_num`."""
        cfg = self._cfg

        ckpt_info = self._load_ckpt(cfg.init_ckpt_path, cfg.ckpt_num)
        step = ckpt_info.get("step", 0)

        Logger.info(f"Run {cfg.num_eval} evaluations at step={step}")
        rollouts, info = self._evaluate(step, cfg.record_video)
        Logger.info(f"Done evaluating {cfg.num_eval} episodes")

        # Save successful terminal states for T-STAR.
        if "episode_success_state" in info.keys():
            success_states = info["episode_success_state"]
            path = Path(cfg.log_dir) / f"success_{step:011d}.pkl"
            Logger.warning(
                f"[*] Store {len(success_states)} successful terminal states: {path}"
            )
            with path.open("wb") as f:
                pickle.dump(success_states, f)

        # Save evaluation statistics.
        info_stat = info.get_stat()
        Path("result").mkdir(exist_ok=True)
        with h5py.File(f"result/{cfg.run_name}.hdf5", "w") as hf:
            for k, v in info.items():
                if np.isscalar(v) or isinstance(v[0], LOG_TYPES):
                    hf.create_dataset(k, data=v)
        with open(f"result/{cfg.run_name}.txt", "w") as f:
            for k, v in info_stat.items():
                f.write(f"{k}\t{v[0]:.03f} $\\pm$ {v[1]:.03f}\n")

        # Record demonstrations.
        if cfg.record_demo:
            demos = [
                dict(obs=v["ob"], actions=v["ac"], rewards=v["rew"], dones=v["done"])
                for v in rollouts
            ]
            fname = f"{cfg.run_name}_step_{step:011d}_{cfg.num_eval}_trajs.pkl"
            path = Path(cfg.demo_dir) / fname
            Logger.warning(f"[*] Generating demo: {path}")
            with path.open("wb") as f:
                pickle.dump(demos, f)

    def _get_agent_by_name(self, algo):
        """Returns RL or IL agent."""
        if algo in RL_ALGOS:
            return RL_ALGOS[algo]
        elif algo in IL_ALGOS:
            return IL_ALGOS[algo]
        else:
            raise ValueError(f"rolf.name={algo} is not supported")

    def _save_ckpt(self, ckpt_num, info=None):
        """Saves checkpoint to log directory.

        Args:
            ckpt_num: number appended to checkpoint name. The number of
                environment step is used in this code.
            info: information required to resume training
        """
        ckpt_path = Path(self._cfg.ckpt_dir) / f"ckpt_{ckpt_num:011d}.pt"
        state_dict = dict(step=ckpt_num, agent=self._agent.state_dict())
        if info:
            state_dict.update(info)
        torch.save(state_dict, ckpt_path)
        Logger.warning(f"[*] Save checkpoint: {ckpt_path}")

        if self._agent.is_off_policy() and self._agent._buffer is not None:
            self._agent.save_replay_buffer(self._cfg.replay_dir, ckpt_num)
        wandb.save(str(ckpt_path))

    def _load_ckpt(self, ckpt_path, ckpt_num):
        """Loads checkpoint with path `ckpt_path` or index number `ckpt_num`.
        If `ckpt_num` is None, it loads the latest checkpoint.
        """
        cfg = self._cfg

        if ckpt_path is None:
            ckpt_path, ckpt_num = get_ckpt_path(cfg.ckpt_dir, ckpt_num)
        else:
            ckpt_num = int(ckpt_path.rsplit("_", 1)[-1].split(".")[0])

        ckpt_info = {}
        if ckpt_path is None:
            Logger.warning("Randomly initialize models")
        else:
            Logger.warning(f"Load checkpoint {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=torch.device(cfg.device))
            self._agent.load_state_dict(ckpt["agent"])

            if cfg.is_train and self._agent.is_off_policy():
                self._agent.load_replay_buffer(cfg.replay_dir, ckpt_num)

            if cfg.init_ckpt_path != ckpt_path or not cfg.init_ckpt_pretrained:
                ckpt_info = {k: v for k, v in ckpt.items() if k != "agent"}
        return ckpt_info

    def _log_train(self, step, train_info, ep_info, name=""):
        """Logs training and episode information to wandb.

        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            ep_info: episode information to log, such as reward, episode time.
            name: postfix for the log section.
        """
        for k, v in train_info.items():
            if np.isscalar(v):
                wandb.log({f"train_rl{name}/{k}": v}, step=step)
            elif np.isscalar(v[0]) or len(v[0].shape) <= 2:
                wandb.log({f"train_rl{name}/{k}": np.mean(v)}, step=step)
            elif len(v[0].shape) == 3:
                wandb.log(
                    {f"train_rl{name}/{k}": [wandb.Image(image) for image in v]},
                    step=step,
                )
            elif len(v[0].shape) >= 4:
                wandb.log(
                    {f"train_rl{name}/{k}": [wandb.Video(video) for video in v]},
                    step=step,
                )

        for k, v in ep_info.items():
            if isinstance(v, LOG_TYPES) or (
                isinstance(v, list) and isinstance(v[0], LOG_TYPES)
            ):
                wandb.log({f"train_ep{name}/{k}": np.mean(v)}, step=step)
                wandb.log({f"train_ep_max{name}/{k}": np.max(v)}, step=step)

    def _log_test(self, step, ep_info, name=""):
        """Logs episode information during testing to wandb.

        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
            name: postfix for the log section.
        """
        if self._cfg.is_train:
            for k, v in ep_info.items():
                if isinstance(v, wandb.Video):
                    wandb.log({f"test_ep{name}/{k}": v}, step=step)
                elif isinstance(v, list) and isinstance(v[0], wandb.Video):
                    for i, video in enumerate(v):
                        wandb.log({f"test_ep{name}/{k}_{i}": video}, step=step)
                elif isinstance(v, list) and isinstance(v[0], LOG_TYPES):
                    wandb.log({f"test_ep{name}/{k}": np.mean(v)}, step=step)
                elif isinstance(v, LOG_TYPES):
                    wandb.log({f"test_ep{name}/{k}": v}, step=step)

    def _update_normalizer(self, rollout):
        """Updates normalizer with `rollout`."""
        if self._cfg.rolf.ob_norm:
            self._agent.update_normalizer(rollout["ob"])

    def _evaluate(self, step=None, record_video=False):
        """Runs `self._cfg.num_eval` rollouts.

        Args:
            step: the number of environment steps.
            record_video: whether to record video or not.
        """
        cfg = self._cfg
        self._agent.eval()
        Logger.info(f"Run {cfg.num_eval} evaluations at step={step}")
        rollouts = []
        info_history = Info()
        for i in range(cfg.num_eval):
            Logger.warning(f"Evalute run {i + 1}")
            rollout, info, frames = self._runner.run_episode(record_video=record_video)
            rollouts.append(rollout)
            check_memory_kill_switch()

            record_video=False
            if record_video:
                rew = info["rew"]
                success = "s" if info.get("episode_success", False) else "f"
                fname = f"{cfg.env.id}_step_{step:011d}_{i}_r_{rew:.3f}_{success}.mp4"
                video_path = self._save_video(fname, frames)
                if cfg.is_train:
                    caption = f"{cfg.run_name}-{step}-{i}"
                    info["video"] = wandb.Video(
                        video_path, caption=caption, fps=15, format="mp4"
                    )

            info_history.add(info)
        self._agent.train()
        return rollouts, info_history

    def _save_video(self, file_name, frames, fps=15.0):
        """Saves `frames` into a video with file name `file_name`."""
        assert not np.issubdtype(frames[0].dtype, np.floating)

        path = Path(self._cfg.video_dir) / file_name
        Logger.warning(f"[*] Generating video with {len(frames)} frames: {path}")
        imageio.mimsave(path, frames, fps=fps)
        Logger.warning(f"[*] Video saved")
        return str(path)
