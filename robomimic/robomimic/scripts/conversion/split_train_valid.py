"""
Helper script to convert a dataset collected using robosuite into an hdf5 compatible with
this repository. Takes a dataset path corresponding to the demo.hdf5 file containing the
demonstrations. It modifies the dataset in-place. By default, the script also creates a
90-10 train-validation split.

For more information on collecting datasets with robosuite, see the code link and documentation
link below.

Code: https://github.com/ARISE-Initiative/robosuite/blob/offline_study/robosuite/scripts/collect_human_demonstrations.py

Documentation: https://robosuite.ai/docs/algorithms/demonstrations.html

Example usage:

    python convert_robosuite.py --dataset /path/to/your/demo.hdf5
"""

import h5py
import json
import argparse

import robomimic.envs.env_base as EB
from robomimic.scripts.split_train_val import split_train_val_from_hdf5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--num_train", type=int, default=None, help="validation ratio, in (0, 1)"
    )
    args = parser.parse_args()

    # split_train_val_from_hdf5(hdf5_path=args.dataset, val_ratio=0.1)
    split_train_val_from_hdf5(hdf5_path=args.dataset, num_train=args.num_train)
