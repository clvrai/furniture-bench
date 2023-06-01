"""Instantiate FurnitureSim-v0 and test various functionalities."""

import argparse
import pickle

import furniture_bench

import gym
import cv2
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Demo path to replay (data directory or pickle)"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="Execute hard-coded assembly script."
    )
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--random-action", action="store_true")
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
    )
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
        "--record", action="store_true", help="Record the video of the simulator."
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input.",
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
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim",
    )

    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    # Create FurnitureSim environment.
    env = gym.make(
        args.env_id,
        furniture=args.furniture,
        num_envs=args.num_envs,
        resize_img=not args.high_res,
        init_assembled=args.init_assembled,
        record=args.record,
        headless=args.headless,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
    )

    # Initialize FurnitureSim.
    ob = env.reset()
    done = False

    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    # Rollout one episode with a selected policy:
    if args.input_device is not None:
        # Teleoperation.
        device_interface = furniture_bench.device.make(args.input_device)

        while not done:
            action, _ = device_interface.get_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)

    elif args.no_action or args.init_assembled:
        # Execute 0 actions.
        while True:
            ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
            ob, rew, done, _ = env.step(ac)

    elif args.random_action:
        # Execute randomly sampled actions.
        import tqdm

        pbar = tqdm.tqdm()
        while True:
            ac = action_tensor(env.action_space.sample())
            ob, rew, done, _ = env.step(ac)
            pbar.update(args.num_envs)

    elif args.file_path is not None:
        # Play actions in the demo.
        with open(args.file_path, "rb") as f:
            data = pickle.load(f)
        for ac in data["actions"]:
            ac = action_tensor(ac)
            env.step(ac)

    elif args.scripted:
        # Execute hard-coded assembly script.
        while not done:
            action, skill_complete = env.get_assembly_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)

    else:
        raise ValueError(f"No action specified")

    print("done")


if __name__ == "__main__":
    main()
