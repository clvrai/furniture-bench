from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.leg import Leg


class DeskLeg(Leg):
    def __init__(self, part_config, part_idx):
        self.half_width = 0.0135
        self.tag_offset = 0.0135

        super().__init__(part_config, part_idx)

        self.reset_x_len = 0.03
        self.reset_y_len = 0.125

        self.reset_gripper_width = 0.06
