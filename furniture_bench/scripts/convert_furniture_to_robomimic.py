# Example command:
# python convert_furniture_to_robosuite.py --data-dir <Path_to_generated_data_directory> --out-file <path_to_hdf5>.hdf5 --next-obs

import os
import argparse
import pickle
import h5py
import json

import numpy as np
import torch
from tqdm import tqdm


import robomimic.utils.tensor_utils as TensorUtils

ENV_NAME = "FurnitureSim-v0"
RANDOMNESS = "low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, required=True, help="directory to pkl dataset"
    )
    parser.add_argument("--out-file", type=str, required=True, help="output file name")
    parser.add_argument("--rot-6d", action="store_true", help="use 6d rotation")
    parser.add_argument("--next-obs", action="store_true", help="add next_obs to data")

    args = parser.parse_args()

    input_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.endswith(".pkl"):
                input_files.append(os.path.join(root, f))
    # Sort by names.
    input_files = sorted(input_files)

    f = h5py.File(args.out_file, "a")  # Write mode

    data = pickle.load(open(input_files[0], "rb"))  # Load first file

    # Create "data" group.
    data_group = f.create_group("data")
    data_group.attrs["env"] = ENV_NAME
    data_group.attrs["env_args"] = json.dumps(
        {
            "furniture": data["furniture"],
            "env_name": ENV_NAME,
            "env_kwargs": {"furniture": data["furniture"]},
        }
    )
    data_group.attrs["env_meta"] = json.dumps(
        {
            "env_name": ENV_NAME,
            "env_kwargs": {"furniture": data["furniture"]},
        }
    )

    data_group.attrs["env_info"] = json.dumps({"env_name": ENV_NAME})

    total_samples = 0
    for demo_idx, input_file in enumerate(tqdm(input_files)):
        with open(input_file, "rb") as f:
            data = pickle.load(f)

            # Create demo group.
            demo_group = data_group.create_group(f"demo_{demo_idx}")
            num_samples = len(data["actions"])  # number of samples in this demo
            demo_group.attrs["num_samples"] = num_samples

            demo_group.create_dataset("actions", data=np.array(data["actions"]))


            demo_group.create_dataset("rewards", data=np.array(data["rewards"]))
            # Create dones data with length of rewards, but last element 1.
            dones = np.concatenate([np.zeros_like(data["rewards"]), np.ones(1)])
            demo_group.create_dataset("dones", data=dones)

            # Extract robot_state keys.
            robot_state_keys = list(data["observations"][0]["robot_state"].keys())
            for obs in data["observations"]:
                for k in robot_state_keys:
                    obs[k] = obs["robot_state"][k]
                del obs["robot_state"]

            obs = TensorUtils.list_of_flat_dict_to_dict_of_list(data["observations"])

            # Change the key of the data["observations"]
            # "parts_poses"   -> "object"
            # "ee_pos"        -> "robot0_eef_pos"
            # "ee_quat"       -> "robot0_eef_quat"
            # "gripper_width" -> "robot0_gripper_qpos"
            # "color_image1"  -> "robot0_eye_in_hand_image"
            # "color_image2"  -> "agentview_image"
            obs["object"] = obs["parts_poses"]
            obs["robot0_eef_pos"] = obs["ee_pos"]
            obs["robot0_eef_quat"] = obs["ee_quat"]
            obs["robot0_gripper_qpos"] = np.array(obs["gripper_width"]).reshape(-1, 1) # Add a dimension since squeezed in data saving.
            obs["robot0_eye_in_hand_image"] = obs["color_image1"]
            obs["agentview_image"] = obs["color_image2"]
            del obs["parts_poses"]
            del obs["ee_pos"]
            del obs["ee_quat"]
            del obs["gripper_width"]
            del obs["color_image1"]
            del obs["color_image2"]

            for k in obs:
                demo_group.create_dataset("obs/{}".format(k), data=np.array(obs[k]))

                if args.next_obs:
                    demo_group.create_dataset(
                        "next_obs/{}".format(k), data=np.array(obs[k])[1:]
                    )

            total_samples += num_samples

    data_group.attrs["total"] = total_samples


if __name__ == "__main__":
    main()
