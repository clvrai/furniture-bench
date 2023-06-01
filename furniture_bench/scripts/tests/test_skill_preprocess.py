"""Test script for preprocessing data for skill bench."""
import subprocess


def main():
    fur2skill = {
        "lamp": 7,
        "square_table": 16,
        "desk": 16,
        "drawer": 8,
        "cabinet": 11,
        "round_table": 8,
        "stool": 11,
        "chair": 17,
        "one_leg": 5,
    }

    cmds = []
    for furniture, num_skills in fur2skill.items():
        for from_skill in range(0, num_skills):
            to_skill = from_skill + 1
            cmds.append(
                f"python furniture_bench/scripts/preprocess_data.py --in-data-path /hdd/furniture_data_low/{furniture} --out-data-path skill_bench/{furniture}_processed_from_{from_skill}_to_{to_skill} --from-skill {from_skill} --to-skill {to_skill}"
            )
            cmds.append(
                f"python furniture_bench/scripts/show_trajectory.py --data-path skill_bench/{furniture}_processed_from_{from_skill}_to_{to_skill}/00000.pkl --channel-first --speed-up 10"
            )

    for cmd in cmds:
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
