"""Convert and preprocess the data for the training."""
import pickle
import argparse
from pathlib import Path

import numpy as np

from furniture_bench.robot.misc import concat_robot_state
from furniture_bench.config import config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--in-data-path", help="Path to directory to load the data", required=True
)
parser.add_argument(
    "--out-data-path", help="Path to directory to save the data", required=True
)
parser.add_argument(
    "--save-last-step",
    action="store_true",
    help="Whether to save last step of the trajectory. (For example, for Off-policy learning)",
)
parser.add_argument(
    "--no-robot-state",
    action="store_true",
    help="Do not use robot state.",
)
parser.add_argument(
    "--success-only",
    action="store_true",
    help="Only use successful trajectories",
)
parser.add_argument(
    "--done-when-assembled",
    action="store_true",
    help="Terminate converting when all the parts are assembled.",
)
parser.add_argument(
    "--phase",
    action="store_true",
    help="Terminate converting when all the parts are assembled.",
)
parser.add_argument("--sum-rew", type=int)
parser.add_argument("--skill-start", type=int)
parser.add_argument("--skill-end", type=int)
parser.add_argument(
    "--use-all-cam", action="store_true", help="Use all of images from three cameras."
)
parser.add_argument(
    "--stack-cam", action="store_true", help="Stack images from three cameras."
)
parser.add_argument(
    "--norm-pos-acts",
    action="store_true",
    help="Do not normalize positional actions. [-1 to 1]",
)
parser.add_argument(
    "--norm-pos-x",
    type=float,
    help="Normalization factor of x position.",
    default=0.1001,
)
parser.add_argument(
    "--norm-pos-y",
    type=float,
    help="Normalization factor of y position.",
    default=0.1001,
)
parser.add_argument(
    "--norm-pos-z",
    type=float,
    help="Normalization factor of z position.",
    default=0.1001,
)
args = parser.parse_args()


def main():
    files = list(Path(args.in_data_path).rglob("*.pkl"))
    if len(files) == 0:
        raise Exception("Data path is empty")

    for i, file in enumerate(sorted(files)):
        print(f"[{i + 1} / {len(files)}] converting {file}")
        with open(file, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(f"Fail to load: {file}")
                continue

            if args.success_only and not data["success"]:
                print(f"Skip failed trajectory: {file}")
                continue
            if len(data["observations"]) == 0:
                print(f"Skip empty trajectory: {file}")
                continue
            new_traj = {}
            new_traj["furniture"] = data["furniture"]
            new_traj["observations"] = data["observations"].copy()
            new_traj["actions"] = data["actions"].copy()
            new_traj["rewards"] = data["rewards"].copy()
            new_traj["skills"] = data["skills"].copy()

            # Skip no action.
            no_action = np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32)
            len_traj = len(data["actions"])
            num_skipped = 0
            for i in range(len_traj):
                if np.isclose(new_traj["actions"][i], no_action).all():
                    num_skipped += 1
                else:
                    break

            new_traj["observations"] = new_traj["observations"][num_skipped:]
            new_traj["actions"] = new_traj["actions"][num_skipped:]
            new_traj["rewards"] = new_traj["rewards"][num_skipped:]
            new_traj["skills"] = new_traj["skills"][num_skipped:]

            print(f"Number of skipped actions: {num_skipped}")

            if args.done_when_assembled:
                sum_rew = 0
                for done_idx, rew in enumerate(new_traj["rewards"]):
                    sum_rew += rew
                    if (
                        sum_rew
                        == config["furniture"][data["furniture"]]["total_reward"]
                    ):
                        break
                done_idx = done_idx + 1 if done_idx + 2 < len_traj else len_traj - 1
                new_traj["observations"] = new_traj["observations"][: done_idx + 1]
                new_traj["actions"] = new_traj["actions"][:done_idx]
                new_traj["rewards"] = new_traj["rewards"][:done_idx]

            if args.skill_start is not None and args.skill_end is not None:
                skill = 0
                skill_start_idx = 0
                for idx, skill_complete in enumerate(new_traj["skills"]):
                    if skill_complete == 1:
                        if idx + 1 < len_traj and new_traj["skills"][idx + 1] == 1:
                            import pdb

                            pdb.set_trace()
                        skill += 1
                        if skill == args.skill_start:
                            skill_start_idx = idx
                        if skill == args.skill_end:
                            skill_done_idx = idx
                done_idx = (
                    skill_done_idx + 1
                    if skill_done_idx + 2 < len_traj
                    else len_traj - 1
                )
                new_traj["observations"] = new_traj["observations"][
                    skill_start_idx : done_idx + 1
                ]
                new_traj["actions"] = new_traj["actions"][skill_start_idx:done_idx]
                new_traj["rewards"] = new_traj["rewards"][skill_start_idx:done_idx]

            if not args.save_last_step:
                new_traj["observations"].pop()

            print(
                f"Number of truncated last steps: {len_traj - (done_idx + 1) if args.done_when_assembled else 0}"
            )
            print(f"Length of new trajectory {len(new_traj['actions'])}")

            # Make it channel first.
            for o in new_traj["observations"]:
                for img in ["color_image1", "color_image2"]:
                    o[img] = np.moveaxis(o[img], -1, 0)

            if args.use_all_cam:
                # Use all three camera input images.
                if args.stack_cam:
                    new_traj["observations"] = [
                        {
                            "image": np.concatenate(
                                [
                                    o["color_image1"],
                                    o["color_image2"],
                                    o["color_image3"],
                                ],
                                axis=0,
                            ),
                            "robot_state": concat_robot_state(o["robot_state"]),
                        }
                        for o in new_traj["observations"]
                    ]
                else:
                    new_traj["observations"] = [
                        {
                            "image1": o["color_image1"],  # Camera 1 image
                            "image2": o["color_image2"],  # Camera 2 image
                            "image3": o["color_image3"],  # Camera 3 image
                            "robot_state": concat_robot_state(o["robot_state"]),
                        }
                        for o in new_traj["observations"]
                    ]
            else:
                new_traj["observations"] = [
                    {
                        "image1": o["color_image1"],  # Wrist cam
                        "image2": o["color_image2"],  # Front cam
                        "robot_state": concat_robot_state(o["robot_state"]),
                    }
                    for o in new_traj["observations"]
                ]

            if np.array(new_traj["rewards"]).sum() <= 0:
                print("Label reward 1")
                new_traj["rewards"][-1] = 1.0

            if args.no_robot_state:
                for obs in new_traj["observations"]:
                    obs.pop("robot_state")

            norm_eps = 1e-5
            cnt = 0
            last_sign = -1

            for i, act in enumerate(new_traj["actions"]):
                if act[6] < 0:
                    act[3:7] = -act[3:7]  # Make sure quaternion scalar is positive.
                if args.norm_pos_acts:
                    act[0] /= args.norm_pos_x
                    act[1] /= args.norm_pos_y
                    act[2] /= args.norm_pos_z
                    act = np.clip(act, -1 + norm_eps, 1 - norm_eps)

                if args.phase:
                    if np.sign(act[7]) != np.sign(last_sign) and cnt < 4:
                        new_traj["rewards"][i] = 1.0
                        last_sign = np.sign(act[7])
                        cnt += 1

        if args.norm_pos_acts:
            print(
                f"Normalization factor: ({args.norm_pos_x}, {args.norm_pos_y}, {args.norm_pos_z})"
            )

        out_dir = Path(args.out_data_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        new_file = out_dir / file.name

        print(f">> save to {new_file}")

        with open(new_file, "wb") as f:
            pickle.dump(new_traj, f)


if __name__ == "__main__":
    main()
