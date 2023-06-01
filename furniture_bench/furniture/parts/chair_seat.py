import numpy as np

import furniture_bench.utils.transform as T
from furniture_bench.utils.pose import cosine_sim, get_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.config import config


class ChairSeat(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        tag_ids = part_config["ids"]
        self.rel_pose_from_center[tag_ids[0]] = get_mat([0, 0, -0.03375], [0, 0, 0])
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [-0.04875, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat([0, 0, 0.065], [0, np.pi, 0])
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [0.04875, 0, 0], [0, -np.pi / 2, 0]
        )

        self.reset_gripper_width = 0.05

        self.reset_x_len = 0.10
        self.reset_y_len = 0.11875

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        return (
            cosine_sim(reset_ori[:3, 2], pose[:3, 2]) > 0.9
            and cosine_sim(reset_ori[:3, 0], pose[:3, 0]) > 0.9
        )

    def randomize_init_pose(self, from_skill=0, pos_range=[-0.05, 0.05], rot_range=45):
        # Too big, so we need to reduce the range.
        if from_skill in [0, 1]:
            pos_range = [-0.05, 0.035]
        super().randomize_init_pose(
            from_skill=from_skill,
            pos_range=pos_range,
            rot_range=rot_range,
        )
