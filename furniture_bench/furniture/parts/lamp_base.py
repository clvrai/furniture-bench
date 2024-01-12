import torch
import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
from furniture_bench.config import config
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class LampBase(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        self.assembled_rel_poses = [get_mat([0, -0.073, 0], [0, 0, np.pi])]

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, 0, -0.0301715], [-0.8726646259971648, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.0301715, 0, 0], [-0.8726646259971648, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0.0, 0, 0.0301715], [-0.8726646259971648, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [-0.0301715, 0, 0], [-0.8726646259971648, np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, 0, 0], [-np.pi / 2, 0, 0]
        )

        self.reset_gripper_width = 0.08
        self.reset_x_len = 0.0847
        self.reset_y_len = 0.0847
        self.base_grip_width = 0.055
        self.half_length = 0.042

        self.skill_complete_next_states = [
            "push_x",
            "go_up",
        ]  # Specificy next state after skill is complete.

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return (
            pose[0, 0] > ori_bound
            or pose[1, 0] > ori_bound
            or pose[0, 0] < -ori_bound
            or pose[1, 0] < -ori_bound
        )

    def reset(self):
        self.pre_assemble_done = False
        self._state = "reach_body_grasp_xy"
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
        base_pose = C.to_homogeneous(
            rb_states[part_idxs["lamp_base"]][0][:3],
            C.quat2mat(rb_states[part_idxs["lamp_base"]][0][3:7]),
        )

        base_pose = sim_to_april_mat @ base_pose
        device = base_pose.device

        if self._state == "reach_body_grasp_xy":
            rot = (
                torch.tensor(rot_mat([-np.pi / 2, 0, np.pi / 2], hom=True))
                .float()
                .to(base_pose.device)
                @ base_pose[:4, :4]
            )
            pos = base_pose[:3, 3] + torch.tensor([0.00, 0.00, 0.03]).float().to(
                base_pose.device
            )
            pos = torch.concat([pos, torch.tensor([1.0]).float().to(base_pose.device)])

            target_pos = (april_to_robot @ pos)[:3]
            target_ori = (april_to_robot @ rot)[:3, :3]
            target_pos[2] = ee_pos[2]
            # target_ori = @ target_ori
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_body_grasp_z"
        elif self._state == "reach_body_grasp_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ base_pose)[2, 3]
            target_pos[2] += 0.04
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)

            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "pick_body"
        elif self._state == "pick_body":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, self.base_grip_width, cnt_max=15):
                self.prev_pose = target
                next_state = "push_x"
        elif (
            self._state == "push_x"
        ):  # Push to the forward so that it doesn't collide with the lamp bulb.
            target_pos = self.prev_pose[:3, 3].clone()
            target_pos[0] += 0.10  # 10cm.
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)

            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "push"
        elif self._state == "push":
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
            target_pos[0] -= self.half_length * 2 + 0.02
            target_pos[1] -= self.half_length + 0.02  # Margin 2cm
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
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
                cnt_max=20,
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
            torch.tensor([self.gripper_action]).to(ee_pos.device),
            skill_complete,
        )
