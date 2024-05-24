import isaacgym
import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import (Batch, D4RLDataset, ReplayBuffer, split_into_trajectories,
                           FurnitureDataset)
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string("save_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
# flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_string("ckpt_step", None, "Specific checkpoint step")
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None, 'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean("red_reward", False, "Use learned reward")
flags.DEFINE_boolean("use_encoder", False, "Use ResNet18 for the image encoder.")
flags.DEFINE_string("encoder_type", '', 'vip or r3m')
flags.DEFINE_boolean('wandb', True, 'Use wandb')
flags.DEFINE_string('wandb_project', 'furniture-bench', 'wandb project')
flags.DEFINE_string('wandb_entity', 'clvr', 'wandb entity')
config_flags.DEFINE_config_file('config',
                                'configs/antmaze_finetune_config.py',
                                'File path to the training hyperparameter configuration.',
                                lock_config=False)

flags.DEFINE_multi_string("data_path", '', "Path to data.")
flags.DEFINE_string('normalization', '', '')
flags.DEFINE_integer('iter_n', -1, 'Reward relabeling iteration')


# REDS
flags.DEFINE_string("task_name", "furniture_one_leg", "Name of task name.")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("image_keys", "color_image2|color_image1", "image keys.")
flags.DEFINE_string("rm_type", "RFE", "reward model type.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions, dataset.rewards,
                                    dataset.masks, dataset.dones_float, dataset.next_observations)

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


def max_normalize(rewards, max_rew):
    """Divide the rewards by the maximum value."""
    normalized_data = np.array([x / max_rew for x in rewards])

    return normalized_data


def replay_chunk_to_seq(trajectories):
    """From: BPref-v2/bpref_v2/utils/reds_extract_reward.py"""
    seq = []

    for i in range(FLAGS.window_size - 1):
        elem = {}
        elem["is_first"] = i == 0
        for key in ["observations", "rewards"]:
            if key == "observations":
                for _key, _val in trajectories[key][0].items():
                    elem[_key] = _val
            elif key == "rewards":
                try:
                    elem["reward"] = trajectories[key][0].squeeze()
                except:
                    elem['reward'] = trajectories[key][0]
            elif isinstance(trajectories[key], np.ndarray):
                elem[key] = trajectories[key][0]
        seq.append(elem)

    for i in range(len(trajectories["observations"])):
        elem = {}
        elem["is_first"] = i == -1
        for key in ["observations", "rewards"]:
            if key == "observations":
                for _key, _val in trajectories[key][i].items():
                    elem[_key] = _val
            elif key == "rewards":
                try:
                    elem["reward"] = trajectories[key][i].squeeze()
                except:
                    elem['reward'] = trajectories[key][i]
            elif isinstance(trajectories[key], np.ndarray):
                elem[key] = trajectories[key][i]
        seq.append(elem)

    return seq


def make_env_and_dataset(env_name: str, seed: int, data_path: str, use_encoder: bool,
                         encoder_type: str, red_reward: bool=False,
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
            squeeze_done_reward=True,
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


    return env, dataset


def main(_):
    root_logdir = os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed))
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.data_path,
                                        FLAGS.use_encoder, FLAGS.encoder_type,
                                        FLAGS.red_reward, FLAGS.iter_n)

    if FLAGS.normalization == "min_max":
        min_max_normalize(dataset)
    if FLAGS.normalization == "max":
        max_rew = np.max(dataset.rewards)
        dataset.rewards = max_normalize(dataset.rewards, max_rew)

    kwargs = dict(FLAGS.config)
    if FLAGS.wandb:
        import wandb
        wandb.tensorboard.patch(root_logdir=root_logdir)
        wandb.init(project=FLAGS.wandb_project,
                   entity=FLAGS.wandb_entity,
                   name=FLAGS.env_name + '-' + str(FLAGS.seed) + '-' + str(FLAGS.run_name) + '-finetune',
                   config=kwargs,
                   sync_tensorboard=True)

    summary_writer = SummaryWriter(root_logdir, write_to_disk=True)

    # agent = Learner(FLAGS.seed,
    #                 env.observation_space.sample()[np.newaxis],
    #                 env.action_space.sample()[np.newaxis], **kwargs)
    agent = Learner(FLAGS.seed,
                env.observation_space.sample(),
                env.action_space.sample()[np.newaxis],
                max_steps=FLAGS.max_steps,
                **kwargs,
                use_encoder=FLAGS.use_encoder)

    if FLAGS.red_reward:
        # load reward model.
        from bpref_v2.reward_model.rfe_reward_model import RFERewardModel
        reward_model = RFERewardModel(
            task=FLAGS.task_name,
            model_name=FLAGS.rm_type,
            rm_path=FLAGS.ckpt_path,
            camera_keys=FLAGS.image_keys.split("|"),
            reward_scale=None,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            reward_model_device=0,
            encoding_minibatch_size=16,
            use_task_reward=True,
            use_scale=False,
        )

    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    agent.load(ckpt_dir, FLAGS.ckpt_step or FLAGS.max_steps)


    eval_returns = []
    observation, done = env.reset(), False
    
    observations = []
    actions = []
    rewards = []
    done_floats = []
    masks = []
    next_observations = []

    for i in tqdm.tqdm(range(FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        while not done:
            obs_without_rgb = {k: v for k, v in observation.items() if k != 'color_image1' and k != 'color_image2'}
            action = agent.sample_actions(obs_without_rgb)
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            done_floats.append(float(done))
            masks.append(mask)
            next_observations.append(next_observation)

            observation = next_observation
        len_curr_traj = len(observations)
        assert len_curr_traj == len(actions) == len(rewards) == len(masks) == len(next_observations)
        assert done_floats[-1] == 1.0
    
        if FLAGS.red_reward:
            # compute reds reward
            x = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
            }
            seq = reward_model(replay_chunk_to_seq(x))
            rewards = np.asarray([elem[reward_model.PUBLIC_LIKELIHOOD_KEY] for elem in seq])
            if FLAGS.normalization == "min_max":
                min_max_normalize(rewards)
            if FLAGS.normalization == "max":
                rewards = max_normalize(rewards, max_rew)
        # Remove RGB from the observations.
        observations = [{k: v for k, v in obs.items() if k != 'color_image1' and k != 'color_image2'}
                        for obs in observations]

        # Append to dataset.
        dataset.add_trajectory(observations, actions, rewards, masks, done_floats, next_observations)

        observations = []
        actions = []
        rewards = []
        done_floats = []
        masks = []
        next_observations = []

        observation, done = env.reset(), False
        # for k, v in info['episode'].items():
        #     summary_writer.add_scalar(f'training/{k}', v, info['total']['timesteps'])
        
        # Update as the length of the current trajectory.
        for update_idx in range(len_curr_traj):
            batch = dataset.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if update_idx % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

    if FLAGS.wandb:
        wandb.finish()


if __name__ == '__main__':
    app.run(main)
