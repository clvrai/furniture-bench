import numpy as np
import numpy.typing as npt

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, is_similar_rot, rot_mat
from furniture_bench.config import config
import furniture_bench.utils.transform as T


class StoolLeg(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, -0.0165], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [-0.0165, 0, 0], [0.2617993877991494, np.pi / 2, 0]
        )  # 15 degree
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [-0.0, 0, 0.0165], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0.0165, 0, 0], [0.2617993877991494, -np.pi / 2, 0]
        )
        # self.center_from_anchor = get_mat([0, 0., 0.0165], [0, 0, 0])
        self.reset_x_len = 0.051077
        self.reset_y_len = 0.0875

        self.part_attached_skill_idx = 4

    def is_in_reset_ori(
        self, pose: npt.NDArray[np.float32], from_skill, ori_bound
    ) -> bool:
        # y-axis of the leg align with y-axis of the base.
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False
