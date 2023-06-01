import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.table_top import TableTop


class SquareTableTop(TableTop):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.half_width = 0.08125
        # self.reset_x_len = 0.08125
        self.reset_x_len = 0.1625
        self.reset_y_len = self.reset_x_len

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, -self.half_width], [0, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [self.half_width, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0, self.half_width], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [-self.half_width, 0, 0], [0, np.pi / 2, 0]
        )
