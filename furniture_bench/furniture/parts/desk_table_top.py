import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.table_top import TableTop


class DeskTableTop(TableTop):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, -0.07375], [0, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.1025, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0, 0.07375], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [-0.1025, 0, 0], [0, np.pi / 2, 0]
        )
        # self.center_from_anchor = get_mat([0, 0, 0.0737], [0, 0, 0])

        self.reset_x_len = 0.21
        self.reset_y_len = 0.15

    def randomize_init_pose(self, from_skill=0, pos_range=[-0.05, 0.05], rot_range=45):
        # Too big, so we need to reduce the range.
        if from_skill in [0, 1]:
            pos_range = [-0.05, 0.02]
        super().randomize_init_pose(
            from_skill=from_skill, pos_range=pos_range, rot_range=rot_range
        )
