import numpy as np
import numpy.typing as npt

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.utils.transform as T


class CabinetTop(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, -0.0275], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.05875, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0.0, 0.0275], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [-0.05875, 0.0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.01375, 0], [-np.pi / 2, 0, 0]
        )

        self.reset_gripper_width = 0.08

        self.reset_x_len = 0.0565
        self.reset_y_len = 0.09525

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[1, 0] > ori_bound or pose[1, 0] <= -ori_bound
