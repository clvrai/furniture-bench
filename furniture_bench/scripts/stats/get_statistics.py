import sys
import argparse
import pickle
from pathlib import Path

from datetime import datetime
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Path to directory to load the data", required=True
    )
    parser.add_argument("--out-file", default="log.txt")
    args = parser.parse_args()
    sys.stdout = open(args.out_file, "a")
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    data_dir = Path(args.data_dir)
    total_minutes = 0
    total_transitions = 0
    total_num_trajs = 0
    for furn in data_dir.iterdir():
        print("---------------------")
        print(f"scanning {furn}")
        files = list(Path(furn).rglob("*.pkl"))
        num_traj = len(files)
        total_num_trajs += num_traj
        print(f"The number of traj: {num_traj}")
        furn_succ = 0
        furn_fail = 0
        furn_len_traj = 0
        furn_error = 0
        furn_min_traj = np.inf
        furn_max_traj = -np.inf
        furn_total_rew = 0
        for i, file in enumerate(sorted(files)):
            with open(file, "rb") as f:
                try:
                    data = pickle.load(f)
                except:
                    print(f"Error in {file}")
                if data["success"] == True:
                    furn_succ += 1
                elif data["success"] == False:
                    furn_fail += 1
                    continue
                else:
                    print(furn, i)
                len_traj = len(data["observations"])
                if len_traj == 0:
                    print(furn, i)
                if "error" not in data:
                    data["error"] = False
                if data["error"] == True:
                    furn_error += 1
                elif data["error"] == False:
                    pass
                else:
                    print(furn, i)
                furn_len_traj += len_traj
                furn_max_traj = max(len_traj, furn_max_traj)
                furn_min_traj = min(len_traj, furn_min_traj)

                for rew in data["rewards"]:
                    furn_total_rew += rew

        print(f"Sum of length of trajectories {furn_len_traj}")
        print(f"Average reward: {furn_total_rew / num_traj}")
        furn_minutes = furn_len_traj * 0.2 / 60
        total_minutes += furn_minutes
        print(f"furniture minutes: {furn_minutes}")
        print(f"furniture hours: {furn_minutes / 60}")
        print(f"Average length of trajectories {furn_len_traj / num_traj}")
        print(f"max length traj: {furn_max_traj}")
        print(f"min length traj: {furn_min_traj}")

        print(f"Num success: {furn_succ}")
        print(f"Num failure: {furn_fail}")

        print(f"Num error: {furn_error}")
        print(f"Average success rate {furn_succ / num_traj}")
        print(f"Average failure rate {furn_fail / num_traj}")
        print("---------------------")
        total_transitions += furn_len_traj

    print(f"totaltransitions: {total_transitions}")
    print(f"total num trajectories: {total_num_trajs}")

    print(f"Total data minutes {total_minutes}")


if __name__ == "__main__":
    main()
