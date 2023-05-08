from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.leg import Leg


class SquareTableLeg(Leg):
    def __init__(self, part_config, part_idx):
        self.tag_offset = 0.015
        self.half_width = 0.015

        self.reset_x_len = 0.03 + 0.02  # 0.02 is the margin
        self.reset_y_len = 0.0875
        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.06
        self.grasp_margin_x = 0
        self.grasp_margin_z = 0
