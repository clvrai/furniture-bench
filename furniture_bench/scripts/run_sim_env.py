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
    parser.add_argument(
        "--replay-path", type=str, help="Path to the saved data to replay action."
    )

    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space.",
        choices=["quat", "axis", "rot_6d"],
        default="quat",
    )

    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )

    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
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
        act_rot_repr=args.act_rot_repr,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
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
            if args.act_rot_repr == "quat":
                ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
            else:
                ac = action_tensor([0, 0, 0, 0, 0, 0, -1])
            ob, rew, done, _ = env.step(ac)
    elif args.random_action:
        # Execute randomly sampled actions.
        from tqdm import tqdm

        episodes = 256
        steps = 300
        from cProfile import Profile
        from pstats import Stats, SortKey
        per_step_times = []
        import time
        # with Profile() as profile:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]

        gpu_memories = [] # MB unit.
        for epi in tqdm(range(int(episodes / args.num_envs))):
            env.reset()
            for i in range(steps):
                ac = action_tensor(env.action_space.sample())
                start = time.time()
                ob, rew, done, _ = env.step(ac)
                end = time.time()
                per_step_times.append(end - start)
                
                if i % 50 == 0:
                    gpu_memories.append(gpu.memoryUsed)

            env.reset()

        from contextlib import redirect_stdout
        from furniture_bench.envs.furniture_sim_env import observation_times, reward_times, control_times, control_refresh_times
        with open(f'sim_speed_{args.num_envs}.txt', 'w') as f:
            with redirect_stdout(f):
                print(f"Task: {args.furniture}", f"observation: parts_poses, color_image1, color_image2")
                print("=======================================================")
                print(f"Average steps per time: {1 / np.mean(per_step_times) * args.num_envs}")
                print(f"Average observation per time: {1 / np.mean(observation_times) * args.num_envs}")
                print(f"Average reward per time: {1 / np.mean(reward_times) * args.num_envs}")
                print(f"Average control per time: {1 / np.mean(control_times) * args.num_envs}")
                print(f"Average control refresh per time: {1 / np.mean(control_refresh_times) * args.num_envs}")

                print(f"Number of steps: {len(per_step_times) * args.num_envs}")
                print(f"Number of observations: {len(observation_times) * args.num_envs}")
                print(f"Number of rewards: {len(reward_times) * args.num_envs}")
                print(f"Number of controls: {len(control_times) * args.num_envs}")
                print(f"Number of controls refresh: {len(control_refresh_times) * args.num_envs}")
                
                print("=======================================================")
                
                print(f"Average GPU memory (MB): {np.mean(gpu_memories)}")

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
    elif args.replay_path:
        # Replay the trajectory.
        with open(args.replay_path, "rb") as f:
            data = pickle.load(f)
        env.reset_to([data["observations"][0]])  # reset to the first observation.
        for ac in data["actions"]:
            ac = action_tensor(ac)
            ob, rew, done, _ = env.step(ac)
    else:
        raise ValueError(f"No action specified")

    print("done")


if __name__ == "__main__":
    main()
