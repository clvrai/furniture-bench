from furniture_bench.utils.pose import cosine_sim
from furniture_bench.furniture.parts.leg import Leg


class RoundTableLeg(Leg):
    def __init__(self, part_config, part_idx):
        self.reset_x_len = 0.0425
        self.reset_y_len = 0.09125

        self.tag_offset = 0.01625
        self.half_width = 0.01625
        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.05
        self.grasp_margin_x = 0.020
        self.grasp_margin_z = 0.020

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        return cosine_sim(reset_ori[:3, 1], pose[:3, 1]) > 0.9
