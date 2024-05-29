import isaacgym
import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import wandb
from einops import rearrange

import wrappers
from dataset_utils import D4RLDataset, FurnitureDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000000, "Eval interval.")
flags.DEFINE_integer("min_eval_step", 600000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Ckpt interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("red_reward", False, "Use learned reward")
# flags.DEFINE_string("data_path", '', "Path to data.")
flags.DEFINE_multi_string("data_path", '', "Path to data.")
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_boolean("use_encoder", False, "Use ResNet18 for the image encoder.")
flags.DEFINE_string("encoder_type", '', 'vip or r3m')
flags.DEFINE_boolean('wandb', False, 'Use wandb')
flags.DEFINE_string('wandb_project', '', 'wandb project')
flags.DEFINE_string('wandb_entity', '', 'wandb entity')
flags.DEFINE_string('normalization', '', '')
flags.DEFINE_integer('iter_n', -1, 'Reward relabeling iteration')
flags.DEFINE_boolean('use_layer_norm', False, 'Use layer normalization')
flags.DEFINE_boolean('phase_reward', False, 'Use phase reward.')

flags.DEFINE_string("opt_decay_schedule", "cosine", "")


def normalize(dataset):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def min_max_normalize(dataset):
    max_val = np.max(dataset.rewards)
    min_val = np.min(dataset.rewards)

    normalized_data = np.array(
        [(x - min_val) / (max_val - min_val) for x in dataset.rewards]
    )
    normalized_data -= 1  # (0, 1) -> (-1, 0)

    dataset.rewards = normalized_data


def max_normalize(dataset):
    """Divide the rewards by the maximum value."""
    max_val = np.max(dataset.rewards)

    normalized_data = np.array([x / max_val for x in dataset.rewards])

    dataset.rewards = normalized_data


def make_env_and_dataset(env_name: str, seed: int, data_path: str, use_encoder: bool,
                         encoder_type: str, red_reward: bool=False,
                         normalization:str = None,
                         iter_n: int = -1) -> Tuple[gym.Env, D4RLDataset]:
    if "Furniture" in env_name:
        import furniture_bench

        env_id, furniture_name = env_name.split("/")
        # env = gym.make(env_id,
        #                furniture=furniture_name,
        #                data_path=data_path,
        #                use_encoder=use_encoder,
        #    encoder_type=encoder_type)
        env = gym.make(
            env_id,
            furniture=furniture_name,
            # max_env_steps=600,
            headless=True,
            num_envs=1,  # Only support 1 for now.
            manual_done=False,
            # resize_img=True,
            # np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
            # channel_first=False,
            randomness="low",
            compute_device_id=0,
            graphics_device_id=0,
            # gripper_pos_control=True,
            encoder_type="r3m",
            phase_reward=FLAGS.phase_reward
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    if "Furniture" in env_name:
        dataset = FurnitureDataset(
            data_path, use_encoder=use_encoder, red_reward=red_reward, iter_n=iter_n
        )
    else:
        dataset = D4RLDataset(env)

    if "antmaze" in env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        normalize(dataset)

    if normalization == "min_max":
        min_max_normalize(dataset)
    if normalization == "max":
        max_normalize(dataset)

    return env, dataset


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    tb_dir = os.path.join(FLAGS.save_dir, "tb", f"{FLAGS.run_name}.{FLAGS.seed}")
    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.data_path,
                                        FLAGS.use_encoder, FLAGS.encoder_type,
                                        FLAGS.red_reward, FLAGS.normalization, FLAGS.iter_n)

    kwargs = dict(FLAGS.config)
    if FLAGS.wandb:
        wandb.init(project=FLAGS.wandb_project,
                   entity=FLAGS.wandb_entity,
                   name=FLAGS.env_name + '-' + str(FLAGS.seed) + '-' + str(FLAGS.run_name),
                   config=kwargs,
                   sync_tensorboard=True)

    summary_writer = SummaryWriter(tb_dir, write_to_disk=True)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample(),
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs,
                    use_encoder=FLAGS.use_encoder,
                    use_layer_norm=FLAGS.use_layer_norm,
                    opt_decay_schedule=FLAGS.opt_decay_schedule, 
                    )
    print(agent)

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", np.array(v), i)
            summary_writer.flush()

        if i > FLAGS.min_eval_step and i % FLAGS.eval_interval == 0:
            # eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            log_video = "WithImage" in FLAGS.env_name
            eval_stats, log_videos = evaluate(agent, env, FLAGS.eval_episodes, log_video=log_video)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
            summary_writer.flush()
            if log_video:
                max_length = max(vid.shape[0] for vid in log_videos)  # Find the maximum sequence length
                padded_vids = np.array([np.pad(vid, ((0, max_length - vid.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant') for vid in log_videos])
                # Make it np.int8
                padded_vids = padded_vids.astype(np.uint8)

                name = "rollout_video"
                fps = 20
                vids = rearrange(padded_vids, 'b t c h w -> (b t) c h w')
                log_dict = {name: wandb.Video(vids, fps=fps, format="mp4")}
                # log_dict = {name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids]}
                wandb.log(log_dict, step=i)

            # eval_returns.append((i, eval_stats["sum_of_reward"]))
            # np.savetxt(
            #     os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"),
            #     eval_returns,
            #     fmt=["%d", "%.1f"],
            # )

        if i % FLAGS.ckpt_interval == 0:
            agent.save(ckpt_dir, i)

    if not i % FLAGS.ckpt_interval == 0:
        # Save last step if it is not saved.
        agent.save(ckpt_dir, i)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
