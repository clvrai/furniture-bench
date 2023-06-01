import numpy as np
import numpy.typing as npt

from furniture_bench.furniture.parts.leg import Leg
from furniture_bench.utils.pose import get_mat, rot_mat


class LampBulb(Leg):
    def __init__(self, part_config, part_idx):
        self.half_width = 0.0175
        self.tag_offset = 0.0175
        self.reset_x_len = 0.057
        self.reset_y_len = 0.13

        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.07
