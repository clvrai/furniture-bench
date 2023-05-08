import argparse
import os

import pandas as pd


def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def main():
    df = pd.DataFrame(columns=["randomness", "furniture", "size (GB)"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-data-path", help="Path to collected data", required=True)

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
            path = os.path.join(args.in_data_path, rand, furn)

            df.loc[len(df)] = [rand, furn, get_dir_size(path) // 10**9]

    df.to_csv("data_size.csv")


if __name__ == "__main__":
    main()
