import os
import subprocess
from typing import Tuple

import gym
import numpy as np

import tqdm
from absl import app, flags
from ml_collections import config_flags
from furniture_bench.utils.checkpoint import download_ckpt_if_not_exists

import wrappers
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_string("ckpt_step", None, "Specific checkpoint step")
flags.DEFINE_string("randomness", "low", "Random mode")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("from_skill", int(0), "Skill to start from.")
flags.DEFINE_integer("skill", int(-1), "Skill to evaluate.")
flags.DEFINE_integer("high_random_idx", int(0), "High random idx.")

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")

flags.DEFINE_boolean("image", True, "Image-based model")
flags.DEFINE_boolean("use_encoder", False, "Use CNN for the image encoder.")
flags.DEFINE_boolean("record", False, "Record video")
flags.DEFINE_string("encoder_type", "r3m", "vip or r3m")
flags.DEFINE_float("temperature", 0.00, "Temperature for the policy.")
flags.DEFINE_boolean('headless', False, 'Run in headless mode')

config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def make_env(
    env_name: str,
    seed: int,
    use_encoder: bool,
    record: bool,
    from_skill: int,
    skill: int,
    randomness: str,
    high_random_idx,
    encoder_type: str,
    headless: bool
) -> gym.Env:
    if "Furniture" in env_name:
        import furniture_bench
        env_id, furniture_name = env_name.split("/")
        env = gym.make(
            env_id,
            furniture=furniture_name,
            use_encoder=use_encoder,
            use_all_cam=False,
            record=record,
            disable_env_checker=True,
            from_skill=from_skill,
            skill=skill,
            high_random_idx=high_random_idx,
            randomness=randomness,
            encoder_type=encoder_type,
            headless=headless
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    return env


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, "eval"), exist_ok=True)
    eval_path = os.path.join(FLAGS.save_dir, "eval", f"{FLAGS.run_name}.{FLAGS.seed}")

    if "Sim" in FLAGS.env_name:
        import isaacgym

    env = make_env(
        FLAGS.env_name,
        FLAGS.seed,
        use_encoder=FLAGS.use_encoder,
        record=FLAGS.record,
        from_skill=FLAGS.from_skill,
        skill=FLAGS.skill,
        high_random_idx=FLAGS.high_random_idx,
        randomness=FLAGS.randomness,
        encoder_type=FLAGS.encoder_type,
        headless=FLAGS.headless
    )

    kwargs = dict(FLAGS.config)
    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        **kwargs,
        use_encoder=FLAGS.use_encoder,
    )

    download_ckpt_if_not_exists(os.path.join(FLAGS.save_dir, "ckpt"), FLAGS.run_name, FLAGS.seed)

    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    agent.load(ckpt_dir, FLAGS.ckpt_step or FLAGS.max_steps)

    eval_stats = evaluate(agent, env, FLAGS.eval_episodes, FLAGS.temperature)
    np.savetxt(eval_path, [eval_stats["return"]], fmt=["%.1f"])


if __name__ == "__main__":
    app.run(main)
