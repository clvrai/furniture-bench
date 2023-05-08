"""Instantiate a FurnitureSimEnv and test various functionalities."""
import argparse
import pickle

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.device import make_device_interface

import gym
import cv2
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Path to directory of data or single pickle file"
    )
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument(
        "--domain_randomization",
        action="store_true",
        help="Whether or not to use domain randomization",
    )
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--init-assembled",
        action="store_true",
        help="Initialize the environment with the assembled furniture.",
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator at the beginning of the episode.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use scripted function to compute actions.",
    )

    parser.add_argument(
        "--resize-img",
        action="store_true",
        help="Whether or not to resize the image observation.",
    )

    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment.",
    )

    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness.",
    )

    parser.add_argument(
        "--from-skill",
        default=0,
        type=int,
        help="Which skill to start from."
    )

    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    env = gym.make(
        "Furniture-Sim-Env-v0",
        furniture=args.furniture,
        with_display=True,
        num_envs=args.num_envs,
        init_assembled=args.init_assembled,
        domain_randomization=args.domain_randomization,
        record=args.record,
        resize_img=args.resize_img,
        headless=args.headless,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
        from_skill=args.from_skill,
    )
    env.reset()

    if args.input_device is not None:
        device_interface = make_device_interface(args.input_device)

        while True:
            action, _ = device_interface.get_action_from_input(use_quat=True)
            obs, r, done, _ = env.step(
                torch.tensor(action).float().to(env.device)[None, :]
            )
            # Single environment.

            k = cv2.waitKey(1)
            if done:
                print("done")
                break
    elif args.no_action or args.init_assembled:
        while True:
            ac = (
                torch.tensor([0, 0, 0, 0, 0, 0, 1, -1])
                .repeat(args.num_envs)
                .reshape(args.num_envs, -1)
                .float()
                .to(env.device)
            )
            _, r, done, _ = env.step(ac)
    elif args.file_path is not None:
        with open(args.file_path, "rb") as f:
            data = pickle.load(f)
        for ac in data["actions"]:
            if env is not None:
                ac = torch.from_numpy(ac).float().to(env.device)[None, :]
                env.step(ac)
    elif args.scripted:  # Run assembly
        done = 0.0
        while not done > 0:
            done = float(env.unwrapped._done().detach().squeeze())
            action, skill_complete = env.get_assembly_action()
            _, done, _, _ = env.step(action)
            if done:
                skill_complete = 1
    else:
        raise ValueError("No action specified")


if __name__ == "__main__":
    main()
