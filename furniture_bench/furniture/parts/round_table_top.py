import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.config import config
import furniture_bench.utils.transform as T


class RoundTableTop(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.reset_x_len = 0.2
        self.reset_y_len = 0.2

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, -0.0625, 0], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [-0.0625, 0, 0], [0, 0, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0.0625, 0], [0, 0, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0.0625, 0, 0], [0, 0, np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.0625, 0.00375], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[5]] = get_mat(
            [-0.0625, 0, 0.00375], [0, np.pi, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[6]] = get_mat(
            [0, 0.0625, 0.00375], [0, np.pi, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[7]] = get_mat(
            [0.0625, 0, 0.00375], [0, np.pi, np.pi / 2]
        )
        # self.center_from_anchor = get_mat([0, 0.0625, 0], [0, 0, 0])
        self.reset_gripper_width = 0.03

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 2] < -ori_bound

    def randomize_init_pose(self, from_skill=0, pos_range=[-0.05, 0.05], rot_range=45):
        # Too big, so we need to reduce the range.
        if from_skill in [0, 1]:
            pos_range = [-0.05, 0.01]
        super().randomize_init_pose(
            from_skill=from_skill, pos_range=pos_range, rot_range=rot_range
        )
