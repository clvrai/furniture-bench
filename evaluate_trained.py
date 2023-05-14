#!/usr/bin/env python3
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Rollout with pre-trained policy")

parser.add_argument(
    "--randomness", type=str, help="the randomness level (low, med, high)"
)
parser.add_argument("--furniture", type=str, help="the furniture type")
parser.add_argument(
    "--is-sim",
    action="store_true",
    help="whether to use simulator or real world environment",
)
parser.add_argument("--encoder_type", type=str, default='r3m',help="r3m or vip")

args = parser.parse_args()

randomness = args.Randomness
furniture = args.Furniture
is_sim = args.is_sim

randomness_options = ["low", "med", "high"]
if randomness not in randomness_options:
    print(f"Unknown randomness argument: {randomness}")
    exit(1)

furnitures = [
    "one_leg",
    "square_table",
    "desk",
    "chair",
    "round_table",
    "lamp",
    "cabinet",
    "drawer",
    "stool",
]
if furniture not in furnitures:
    print(f"Unknown furniture argument: {furniture}")
    exit(1)

print(f"Running {randomness} randomness")

env_name = (
    "Furniture-Image-Feature-Sim-v0/" + furniture
    if is_sim
    else "Furniture-Image-Feature-v0/" + furniture
)

command = [
    "python",
    "implicit_q_learning/test_offline.py",
    "--env_name,",
    env_name,
    "--config=implicit_q_learning/configs/furniture_config.py",
    "--ckpt_step=1000000",
    "--run_name",
    furniture + "_full_r3m_1000",
    "--randomness",
    randomness,
    "--encoder_type",
    args.encoder_type,
]

subprocess.run(command)
