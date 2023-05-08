"""Extract pickl files only from data directory"""
import pickle
import argparse
import os
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-data-path", help="Path to collected data", required=True)
    parser.add_argument("--out-data-path", help="Path to collected data", required=True)

    args = parser.parse_args()

    randomness = os.listdir(args.in_data_path)

    for rand in randomness:
        if rand[0] == ".":
            # Ignore hidden files
            continue
        furnitures = os.listdir(os.path.join(args.in_data_path, rand))
        for furn in furnitures:
            if furn[0] == ".":
                # Ignore hidden files
                continue
            files = list(
                Path(os.path.join(args.in_data_path, rand, furn)).rglob("*.pkl")
            )

            for i, file in enumerate(sorted(files)):
                print(f"[{i + 1} / {len(files)}] converting {file}")
                with open(file, "rb") as f:
                    data = pickle.load(f)

                new_traj = {}
                new_traj["furniture"] = data["furniture"]
                new_traj["observations"] = data["observations"].copy()
                new_traj["actions"] = data["actions"].copy()
                new_traj["rewards"] = data["rewards"].copy()
                new_traj["skills"] = data["skills"].copy()

                new_traj["observations"] = [
                    {
                        "color_image1": o["color_image1"],  # Wrist cam
                        "color_image2": o["color_image2"],  # Front cam
                        "robot_state": o["robot_state"],
                    }
                    for o in new_traj["observations"]
                ]

                out_dir = Path(os.path.join(args.out_data_path, rand, furn))
                out_dir.mkdir(parents=True, exist_ok=True)
                new_file = out_dir / "{:05d}.pkl".format(i)

                print(f">> save to {new_file}")

                with open(new_file, "wb") as f:
                    pickle.dump(new_traj, f)


if __name__ == "__main__":
    main()
