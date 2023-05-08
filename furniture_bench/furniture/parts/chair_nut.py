import numpy as np

from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, rot_mat
import furniture_bench.utils.transform as T


class ChairNut(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        tag_ids = part_config["ids"]
        self.assembled_rel_poses = [
            get_mat((-0.035, 0.08, 0), (np.pi / 2, 0, 0)),
            get_mat((0.035, 0.08, 0), (np.pi / 2, 0, 0)),
        ]

        self.rel_pose_from_center[tag_ids[0]] = get_mat([0, 0, -0.01375], [0, 0, 0])
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [0.01375, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat(
            [0.0, 0.0, 0.01375], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [-0.01375, 0.0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[4]] = get_mat(
            [0.0, 0.0, 0], [-np.pi / 2, 0, 0]
        )
        self.reset_gripper_width = 0.05

        self.reset_x_len = 0.03
        self.reset_y_len = 0.03

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 1] > ori_bound and (
            pose[0, 0] > ori_bound
            or pose[1, 0] > ori_bound
            or pose[0, 0] < -ori_bound
            or pose[1, 0] < -ori_bound
        )
