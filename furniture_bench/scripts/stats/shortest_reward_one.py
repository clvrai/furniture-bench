import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Path to directory to load the data", required=True
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for furn in data_dir.iterdir():
        print(f"scanning {furn}")
        files = list(Path(furn).glob("*.pkl"))

        shot_traj = np.inf
        for file in files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            for obs, _ in data["trajectory"]:
                if obs is None:
                    # Error happened.
                    break
                if "reward" not in obs:
                    continue
                if obs["reward"] == 1.0:
                    shot_traj = min(shot_traj, len(data["trajectory"]))
        print(f"{furn} shortest trajectory for single reward: {shot_traj}")


if __name__ == "__main__":
    main()
