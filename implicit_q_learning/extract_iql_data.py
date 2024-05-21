import pickle
from pathlib import Path

import furniture_bench

import numpy as np
import torch
import gym
from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string("furniture", None, "Furniture name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("out_file_path", None, "Path to save converted data.")


def main(_):
    # if FLAGS.num_threads > 0:
    #     print(f"Setting torch.num_threads to {FLAGS.num_threads}")
    #     torch.set_num_threads(FLAGS.num_threads)

    env_type = "Image"
    env_id = f"Furniture-{env_type}-Dummy-v0"
    furniture = FLAGS.furniture
    demo_dir = FLAGS.demo_dir

    dir_path = Path(demo_dir)

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    red_reward_ = []
    done_ = []

    files = list(dir_path.glob("*.pkl"))
    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    for i, file_path in enumerate(files):
        # if FLAGS.num_demos and i == FLAGS.num_demos:
        #     break
        print(f"Loading [{i+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)

            if len(x["observations"]) == len(x["actions"]):
                # Dummy
                x["observations"].append(x["observations"][-1])
            l = len(x["observations"])

            for i in range(l - 1):
                obs_.append(
                    {
                        # 'image_feature': feature1,
                        "image1": x["observations"][i]["image1"],
                        "image2": x["observations"][i]["image2"],
                        "robot_state": x["observations"][i]["robot_state"],
                    }
                )
                next_obs_.append(
                    {
                        # 'image_feature': next_feature1,
                        "image1": x["observations"][i + 1]["image1"],
                        "image2": x["observations"][i + 1]["image2"],
                        "robot_state": x["observations"][i + 1]["robot_state"],
                    }
                )

                action_.append(x["actions"][i])
                reward_.append(x["rewards"][i])
                if "reds_rewards" in x:
                    red_reward_.append(x["reds_rewards"][i])
                done_.append(1 if i == l - 2 else 0)

    dataset = {
        "observations": obs_,
        "actions": np.array(action_),
        "next_observations": next_obs_,
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }
    if len(red_reward_) > 0:
        dataset["red_rewards"] = np.array(red_reward_)

    path = (
        f"data/{env_type}/{furniture}.pkl"
        if FLAGS.out_file_path is None
        else FLAGS.out_file_path
    )
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with Path(path).open("wb") as f:
        pickle.dump(dataset, f)
        print(f"Saved at {path}")


if __name__ == "__main__":
    app.run(main)
