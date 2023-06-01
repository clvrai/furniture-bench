import numpy as np

from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat
import furniture_bench.utils.transform as T


class LampHood(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.assembled_rel_poses = [
            get_mat([0, 0.094, 0], [0, 0, 0]),
            get_mat([0, 0.085, 0], [0, 0, 0]),
            get_mat([0, 0.09, 0], [0, 0, 0]),
        ]
        self.reset_x_len = 0.088
        self.reset_y_len = 0.088

        # 0.139626 radian = 8 degree.
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, -0.034022], [-0.139626, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, -np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -np.pi / 3, 0],
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -2 * np.pi / 3, 0],
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0, 0, 0.034022], [-0.139626, -3 * np.pi / 3, 0]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -4 * np.pi / 3, 0],
        )
        self.rel_pose_from_center[self.tag_ids[5]] = get_mat(
            [np.cos(np.pi / 3) * 0.034022, 0, -np.sin(np.pi / 6) * 0.034022],
            [-0.139626, -5 * np.pi / 3, 0],
        )

        self.reset_gripper_width = 0.045

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        # Veritical orientation.
        return pose[2, 1] > 0.8
