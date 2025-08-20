import copy
import pdb
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from furniture_bench.furniture.parts.pose_filter import PoseFilter
from furniture_bench.utils.pose import get_mat, is_similar_pos, is_similar_pose, rot_mat
from furniture_bench.utils.pose import is_similar_rot
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class Part(ABC):
    @abstractmethod
    def __init__(self, part_config, part_idx: int):
        # Three pose filter. (Each camera has filter.)
        self.pose_filter = [PoseFilter(), PoseFilter(), PoseFilter()]
        self.part_config = copy.deepcopy(part_config)
        self.name = part_config["name"]
        self.asset_file = part_config["asset_file"]
        self.tag_ids = part_config["ids"]
        self.reset_pos = part_config["reset_pos"].copy()
        self.reset_ori = part_config.get("reset_ori").copy()
        self.center_from_anchor = None  # should be set in subclass.
        self.rel_pose_from_center = {}  # should be set in subclass.
        self.reset_gripper_width = None  # should be set in subclass.
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, 0], [0, 0, 0]
        )  # Anchor tag.

        self.part_idx = part_idx
        self.pre_assemble_done = True
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.2
        self.gripper_action = -1

        self.default_assembled_pose = part_config.get("default_assembled_pose", None)
        self.collision_margin = 0.01
        self.first_setting_target = True
        self.target = None
        self.prev_cnt = 0
        self.curr_cnt = 0
        self.part_moved_skill_idx = part_config.get("part_moved_skill_idx", np.inf)
        self.part_attached_skill_idx = part_config.get(
            "part_attached_skill_idx", np.inf
        )

    def randomize_init_pose(self, from_skill=0, pos_range=[-0.05, 0.05], rot_range=45, generator: Optional[np.random.Generator] = None):
        if generator is None:
            generator = np.random.default_rng()
        self.reset_pos[from_skill][:2] = self.part_config["reset_pos"][from_skill][
            :2
        ] + generator.uniform(
            low=pos_range[0], high=pos_range[1], size=2
        )  # x, y
        self.mut_ori = rot_mat(
            [0, 0, generator.uniform(low=np.radians(-rot_range), high=np.radians(rot_range))],
            hom=True,
        )
        self.reset_ori[from_skill] = (
            self.mut_ori @ self.part_config["reset_ori"][from_skill]
        )

    def randomize_init_pose_high(self, high_random_idx: int):
        self.reset_pos[0] = self.part_config["high_rand_reset_pos"][high_random_idx][0]
        self.reset_ori[0] = self.part_config["high_rand_reset_ori"][high_random_idx][0]

    def is_collision(self, part2):
        """Check if the part is collided with another part without considering rotation."""
        p_x1 = -(self.reset_x_len / 2)
        p_y1 = self.reset_y_len / 2
        p1_p1 = self.mut_ori @ np.array([p_x1, p_y1, 0, 1])

        p_x2 = self.reset_x_len / 2
        p_y2 = self.reset_y_len / 2
        p1_p2 = self.mut_ori @ np.array([p_x2, p_y2, 0, 1])

        p_x3 = -(self.reset_x_len / 2)
        p_y3 = -(self.reset_y_len / 2)
        p1_p3 = self.mut_ori @ np.array([p_x3, p_y3, 0, 1])

        p_x4 = self.reset_x_len / 2
        p_y4 = -(self.reset_y_len / 2)
        p1_p4 = self.mut_ori @ np.array([p_x4, p_y4, 0, 1])

        part2_x1 = -(part2.reset_x_len / 2)
        part2_y1 = part2.reset_y_len / 2
        p2_p1 = part2.mut_ori @ np.array([part2_x1, part2_y1, 0, 1])

        part2_x2 = part2.reset_x_len / 2
        part2_y2 = part2.reset_y_len / 2
        p2_p2 = part2.mut_ori @ np.array([part2_x2, part2_y2, 0, 1])

        part2_x3 = -(part2.reset_x_len / 2)
        part2_y3 = -(part2.reset_y_len / 2)
        p2_p3 = part2.mut_ori @ np.array([part2_x3, part2_y3, 0, 1])

        part2_x4 = part2.reset_x_len / 2
        part2_y4 = -(part2.reset_y_len / 2)
        p2_p4 = part2.mut_ori @ np.array([part2_x4, part2_y4, 0, 1])

        try:
            part1_x1 = min(p1_p1[0], p1_p2[0], p1_p3[0], p1_p4[0])
            part1_x2 = max(p1_p1[0], p1_p2[0], p1_p3[0], p1_p4[0])
            part1_y1 = min(p1_p1[1], p1_p2[1], p1_p3[1], p1_p4[1])
            part1_y2 = max(p1_p1[1], p1_p2[1], p1_p3[1], p1_p4[1])

            part1_x1 = self.reset_pos[0][0] + part1_x1
            part1_x2 = self.reset_pos[0][0] + part1_x2
            part1_y1 = self.reset_pos[0][1] + part1_y1
            part1_y2 = self.reset_pos[0][1] + part1_y2

            part2_x1 = min(p2_p1[0], p2_p2[0], p2_p3[0], p2_p4[0])
            part2_x2 = max(p2_p1[0], p2_p2[0], p2_p3[0], p2_p4[0])
            part2_y1 = min(p2_p1[1], p2_p2[1], p2_p3[1], p2_p4[1])
            part2_y2 = max(p2_p1[1], p2_p2[1], p2_p3[1], p2_p4[1])

            part2_x1 = part2.reset_pos[0][0] + part2_x1
            part2_x2 = part2.reset_pos[0][0] + part2_x2
            part2_y1 = part2.reset_pos[0][1] + part2_y1
            part2_y2 = part2.reset_pos[0][1] + part2_y2
        except:
            pdb.set_trace()

        if (
            part1_x1 > part2_x2 + self.collision_margin
            or part1_x2 < part2_x1 - self.collision_margin
        ):
            return False
        if (
            part1_y1 > part2_y2 + self.collision_margin
            or part1_y2 < part2_y1 - self.collision_margin
        ):
            return False
        return True

    def in_boundary(self, pos_lim, from_skill):
        if (
            self.reset_pos[from_skill][0] < pos_lim[0][0]
            or self.reset_pos[from_skill][0] > pos_lim[0][1]
        ):
            return False
        if (
            self.reset_pos[from_skill][1] < pos_lim[1][0]
            or self.reset_pos[from_skill][1] > pos_lim[1][1]
        ):
            return False
        return True

    def reset_pose_filters(self):
        for pose_filter in self.pose_filter:
            pose_filter.reset()

    def is_in_reset_ori(
        self, pose: npt.NDArray[np.float32], from_skill: int, ori_bound: float
    ) -> bool:
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
            return True
        return False

    def is_in_reset_pose(self, pose, from_skill, pos_threshold, ori_bound):
        if self.is_in_reset_pos(
            pose, from_skill, pos_threshold
        ) and self.is_in_reset_ori(pose, from_skill, ori_bound):
            return True
        print(
            f"[reset] Part {self.__class__.__name__} [{self.part_idx}] is not in the reset pose."
        )

        if not self.is_in_reset_pos(pose, from_skill, pos_threshold):
            print(
                "xy should be ({0:0.3f}, {1:0.3f}), but got ({2:0.3f}, {3:0.3f})".format(
                    self.reset_pos[from_skill][0],
                    self.reset_pos[from_skill][1],
                    pose[0, 3],
                    pose[1, 3],
                )
            )
            return False
        if not self.is_in_reset_ori(pose, from_skill, ori_bound):
            print("Reset orientation mismatch.")
            return False

    def is_in_reset_pos(self, pose, from_skill, pos_threshold):
        """check whether (x, y) position is in reset position."""
        reset_pos = self.reset_pos[from_skill][:2]
        part_pos = np.array(reset_pos)
        detected_pos = np.array(pose[:2, 3])
        return is_similar_pos(
            part_pos[:2], detected_pos[:2], pos_threshold=pos_threshold
        )

    def assemble_done(self, rel_pose, assembled_rel_poses):
        for assembled_rel_pose in assembled_rel_poses:
            if is_similar_pose(
                assembled_rel_pose,
                rel_pose,
                ori_bound=0.96,
                pos_threshold=[0.005, 0.005, 0.005],
            ):
                return True
        return False

    def satisfy(
        self,
        current,
        target,
        pos_error_threshold=None,
        ori_error_threshold=None,
        max_len=25,
    ) -> bool:
        if pos_error_threshold is None:
            pos_error_threshold = self.pos_error_threshold
        if ori_error_threshold is None:
            ori_error_threshold = self.ori_error_threshold

        if ((current[:3, 3] - target[:3, 3]).abs().sum() < pos_error_threshold) and (
            (target[:3, :3] - current[:3, :3]).abs().sum() < ori_error_threshold
        ):
            return True
        if self.curr_cnt - self.prev_cnt >= max_len:
            print("phase time out")
            return True
        return False

    def gripper_less(self, gripper_width, target_width, cnt_max=10):
        if gripper_width <= target_width:
            return True
        if self.curr_cnt - self.prev_cnt >= cnt_max:
            return True
        return False

    def gripper_greater(self, gripper_width, target_width, cnt_max=10):
        if gripper_width >= target_width:
            return True
        if self.curr_cnt - self.prev_cnt >= cnt_max:
            return True
        return False

    def may_transit_state(self, next_state):
        skill_complete = 0
        if next_state != self._state:
            print(f"Changing state from {self._state} to {next_state}")
            self._state = next_state
            if next_state in self.skill_complete_next_states:
                skill_complete = 1
            self.first_setting_target = True
            self.prev_cnt = self.curr_cnt
        self.curr_cnt += 1
        return skill_complete

    def add_noise_first_target(self, target, pos_noise=None, ori_noise=None):
        if self.state_no_noise():
            return target
        if self.first_setting_target:
            if pos_noise is not None:
                target[:3, 3] += pos_noise
            else:
                target[:3, 3] += torch.normal(
                    mean=torch.zeros((3,)), std=torch.ones((3,)) * 0.003
                ).to(target.device)
            ori = C.mat2quat(target[:3, :3]).to(target.device)
            if ori_noise is not None:
                ori = C.quat_multiply(ori, ori_noise).to(target.device)
            else:
                ori = C.quat_multiply(
                    ori,
                    torch.tensor(
                        T.axisangle2quat(
                            [
                                np.radians(np.random.normal(0, 3)),
                                np.radians(np.random.normal(0, 3)),
                                np.radians(np.random.normal(0, 3)),
                            ]
                        ),
                        device=target.device,
                    ),
                ).to(target.device)
            self.target = C.to_homogeneous(target[:3, 3], C.quat2mat(ori))
            self.first_setting_target = False
        return self.target

    def reset(self):
        self.pre_assemble_done = False
        self._state = ""
        self.gripper_action = -1
        self.prev_cnt = 0
        self.curr_cnt = 0
        self.first_setting_target = True

    def state_no_noise(self):
        return False
