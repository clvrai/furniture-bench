import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.utils.transform as T
from furniture_bench.config import config


class CabinetDoor(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, 0], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat([0, 0.07375, 0], [0, 0, 0])
        # self.center_from_anchor = get_mat([0, 0, 0], [0, 0, 0])

        self.reset_gripper_width = 0.04
        self.reset_x_len = 0.054233
        self.reset_y_len = 0.13

        self.part_attached_skill_idx = 4
