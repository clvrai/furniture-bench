"""

Example usage:
    python furniture_bench/scripts/extract_absolute_6d_actions.py --dataset one_leg_300.hdf5
"""
import h5py
import argparse
import numpy as np

import torch
from tqdm import tqdm

import furniture_bench.controllers.control_utils as C
import pytorch3d.transforms as pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r+") as f:
        # Extract absolute actions.
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x[5:]))
        for ep in tqdm(demos):
            actions = f[f"data/{ep}/actions"]
            actions = np.array(actions)

            robot_pos = f[f"data/{ep}/obs/robot0_eef_pos"]
            robot_pos = np.array(robot_pos)

            robot_quat = f[f"data/{ep}/obs/robot0_eef_quat"]
            robot_quat = torch.tensor(np.array(robot_quat))

            absolute_pos_action = robot_pos[:-1] + actions[:, :3]

            absolute_quat_action = np.zeros_like(actions[:, 3:7])
            for i in range(robot_quat[:-1].shape[0]):
                absolute_quat_action[i] = C.quat_multiply(
                    torch.tensor(robot_quat[i]), torch.tensor(actions[i, 3:7])
                )

            # Sanity check.
            for i in range(robot_quat[:-1].shape[0]):
                robot_mat = C.quat2mat(robot_quat[i])
                action_mat = C.quat2mat(torch.tensor(actions[i, 3:7]))
                absolute_mat = C.quat2mat(torch.tensor(absolute_quat_action[i]))

                if not np.allclose(
                    absolute_mat, robot_mat @ action_mat, atol=1e-3, rtol=1e-3
                ):
                    import pdb
                    pdb.set_trace()

            assert absolute_quat_action.sum(axis=1).sum() != 0

            absolute_actions = np.concatenate(
                [absolute_pos_action, absolute_quat_action, actions[:, 7:]], axis=1
            )

            f.create_dataset(f"data/{ep}/absolute_actions", data=absolute_actions)
            f.create_dataset(f"data/{ep}/action_dict/abs_pos", data=absolute_pos_action)
            f.create_dataset(f"data/{ep}/action_dict/gripper", data=actions[:, 7:])

        demos = sorted(list(f["data"].keys()), key=lambda x: int(x[5:]))
        for ep in tqdm(demos):
            for data_path in [f"data/{ep}/absolute_actions", f"data/{ep}/actions"]:
                actions = f[data_path]

                # Create "actions" dataset.
                actions = torch.tensor(np.array(actions))
                org_actions = actions.clone()
                rot_quat = actions[:, 3:7]
                # To rotation matrix.
                rot_mat = pt.quaternion_to_matrix(rot_quat)
                rot_6d = pt.matrix_to_rotation_6d(rot_mat)
                actions = torch.cat([actions[:, :3], rot_6d, actions[:, 7:]], dim=1)
                actions = actions.numpy()

                # Make sure data is converted correctly.
                actions2 = torch.tensor(actions)
                rot_6d2 = actions2[:, 3:9]
                rot_mat2 = pt.rotation_6d_to_matrix(rot_6d2)
                rot_quat2 = pt.matrix_to_quaternion(rot_mat2)
                actions2 = torch.cat(
                    [actions2[:, :3], rot_quat2, actions2[:, 9:]], dim=1
                )
                actions2 = actions2.numpy()
                if not np.allclose(np.abs(org_actions), np.abs(actions2), atol=1e-06):
                    print("max difference:", (np.abs(org_actions) - np.abs(actions2)).max())
                    import pdb

                    pdb.set_trace()

                if data_path == f"data/{ep}/absolute_actions":
                    action_path = f"data/{ep}/absolute_actions_6d"
                    f.create_dataset(f"data/{ep}/action_dict/abs_rot_6d", data=rot_6d.numpy())
                else:
                    action_path = f"data/{ep}/actions_6d"

                # Create the final dataset
                f.create_dataset(action_path, data=actions)
