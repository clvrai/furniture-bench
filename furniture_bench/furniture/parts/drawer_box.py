import numpy as np
from numpy.linalg import inv

import furniture_bench.utils.transform as T
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.config import config


class DrawerBox(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.reset_x_len = 0.11250
        self.reset_y_len = 0.09750
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, -0.0485], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.055, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [-0.055, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0, -0.066375, 0.047], [0, np.pi, 0]
        )
        #                                                          [-np.pi / 2, 0, 0])
        self.reset_gripper_width = 0.06
