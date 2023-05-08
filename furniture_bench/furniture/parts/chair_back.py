import numpy as np

from furniture_bench.utils.pose import cosine_sim, get_mat
from furniture_bench.furniture.parts.part import Part


class ChairBack(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        tag_ids = part_config["ids"]
        self.assembled_rel_poses = [get_mat((0, 0.03, -0.04), (0, 0, 0))]

        self.rel_pose_from_center[tag_ids[0]] = get_mat([0, -0.085, 0], [0, 0, 0])
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [-0.035625, 0, -0.00875], [0, 0, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat(
            [0.035625, 0, -0.00875], [0, 0, 0]
        )
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [0, -0.085, 0.010], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[4]] = get_mat(
            [-0.035625, 0, 0.01875], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[5]] = get_mat(
            [0.035625, 0, 0.01875], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[6]] = get_mat(
            [-0.0488, 0, 0.005], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[7]] = get_mat(
            [0.0488, 0, 0.005], [0, -np.pi / 2, 0]
        )
        # self.center_from_anchor = get_mat([0, 0.085, 0], [0, 0, 0])

        self.reset_gripper_width = 0.06

        self.reset_x_len = 0.10
        self.reset_y_len = 0.21

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        return (
            cosine_sim(reset_ori[:3, 1], pose[:3, 1]) > 0.9
            and cosine_sim(reset_ori[:3, 0], pose[:3, 0]) > 0.9
        )
