import torch
import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.config import config
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class RoundTableTop(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.reset_x_len = 0.2
        self.reset_y_len = 0.2

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, -0.0625, 0], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [-0.0625, 0, 0], [0, 0, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0.0625, 0], [0, 0, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0.0625, 0, 0], [0, 0, np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.0625, 0.00375], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[5]] = get_mat(
            [-0.0625, 0, 0.00375], [0, np.pi, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[6]] = get_mat(
            [0, 0.0625, 0.00375], [0, np.pi, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[7]] = get_mat(
            [0.0625, 0, 0.00375], [0, np.pi, np.pi / 2]
        )
        # self.center_from_anchor = get_mat([0, 0.0625, 0], [0, 0, 0])
        self.reset_gripper_width = 0.03
        self.top_grip_width = 0.043
        self.radius = 0.1

        self.skill_complete_next_states = [
            "push",
            "go_up",
        ]  # Specificy next state after skill is complete.

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 2] < -ori_bound

    def randomize_init_pose(self, from_skill=0, pos_range=[-0.05, 0.05], rot_range=45):
        # Too big, so we need to reduce the range.
        if from_skill in [0, 1]:
            pos_range = [-0.05, 0.01]
        super().randomize_init_pose(
            from_skill=from_skill, pos_range=pos_range, rot_range=rot_range
        )

    def reset(self):
        self.pre_assemble_done = False
        self._state = "reach_top_grasp_xy"
        self.gripper_action = -1

    def pre_assemble(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
    ):
        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        top_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        top_pose = sim_to_april_mat @ top_pose
        device = top_pose.device

        if self._state == "reach_top_grasp_xy":
            # Look at the front
            target_ori = (torch.tensor(rot_mat([np.pi, 0, 0], hom=True)).float())[
                :3, :3
            ]
            pos = top_pose[:3, 3]
            pos = torch.concat([pos, torch.tensor([1.0], device=device)])

            target_pos = (april_to_robot @ pos)[:3]
            target_pos[2] = ee_pos[2]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_top_grasp_z"
        if self._state == "reach_top_grasp_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ top_pose)[2, 3]
            target_pos[2] += 0.03  # Margin.
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori).to(device)

            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                self.gripper_action = 1
                next_state = "pick_top"
        if self._state == "pick_top":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, self.top_grip_width):
                self.prev_pose = target
                next_state = "push"
        if self._state == "push":
            target_pos = torch.zeros((4,), device=device)
            target_pos[-1] = 1
            for name in ["obstacle_front", "obstacle_right", "obstacle_left"]:
                obstacle_pos = torch.cat(
                    [
                        rb_states[part_idxs[name]][0][:3],
                        torch.tensor([1.0], device=device),
                    ]
                )
                target_pos[0] = max(obstacle_pos[0], target_pos[0])
                target_pos[1] = max(obstacle_pos[1], target_pos[1])
            target_pos = april_to_robot @ sim_to_april_mat @ target_pos
            target_pos[0] -= self.radius
            target_pos[1] -= self.radius
            # Margin
            target_pos[2] = ee_pose[2, 3]  # Keep z the same.
            target_pos = target_pos[:3]
            target_ori = self.prev_pose[:3, :3]

            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori),
                pos_noise=torch.normal(
                    mean=torch.zeros((3,)), std=torch.tensor([0.005, 0.005, 0.0])
                ).to(device),
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.5
            ):
                self.prev_pose = target
                self.gripper_action = -1
                next_state = "release"
        if self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["round_table"] - 0.001,
            ):
                next_state = "go_up"
        if self._state == "go_up":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = 0.1
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "done"
        if self._state == "done":
            self.gripper_action = -1
            self.pre_assemble_done = True
            target = self.prev_pose

        skill_complete = self.may_transit_state(next_state)
        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )
