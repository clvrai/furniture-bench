import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.config import config
import furniture_bench.utils.transform as T


class StoolSeat(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, -0.048], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.048, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0, 0.048], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [-0.048, 0, 0], [0, np.pi / 2, 0]
        )
        # self.center_from_anchor = get_mat([0, 0, 0.048], [0, 0, 0])
        self.reset_x_len = 0.0875
        self.reset_y_len = 0.0875

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 1] < -ori_bound
