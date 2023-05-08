import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.config import config
import furniture_bench.utils.transform as T


class DrawerContainer(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        self.reset_x_len = 0.10275
        self.reset_y_len = 0.0825

        tag_ids = part_config["ids"]
        self.rel_pose_from_center[tag_ids[0]] = get_mat([0, 0.0, -0.040], [0, 0, 0])
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [0.050125, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat(
            [-0.050125, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [-0.0375, 0, 0.0405], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[4]] = get_mat(
            [0.0375, 0, 0.0405], [0, np.pi, 0]
        )

        self.reset_gripper_width = 0.035
        self.part_attached_skill_idx = 4
