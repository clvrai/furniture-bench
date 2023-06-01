import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.leg import Leg


class ChairLeg(Leg):
    def __init__(self, part_config, part_idx):
        self.half_width = 0.01375
        self.tag_offset = 0.01375

        self.reset_x_len = 0.03
        self.reset_y_len = 0.085

        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.06
