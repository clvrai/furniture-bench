import torch
import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.utils.transform as T
from furniture_bench.config import config
import furniture_bench.controllers.control_utils as C


class CabinetDoor(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, 0], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat([0, 0.07375, 0], [0, 0, 0])
        # self.center_from_anchor = get_mat([0, 0, 0], [0, 0, 0])

        self.reset_gripper_width = 0.04
        self.reset_x_len = 0.054233
        self.reset_y_len = 0.13

        self.part_attached_skill_idx = 4

        self.grasp_margin_x = 0
        self.grasp_margin_z = 0

        self.skill_complete_next_states = [
            "lift_up",
            "done",
        ]  # Specificy next state after skill is complete. Screw done is handle in `get_assembly_action`

    def reset(self):
        self.prev_pose = None
        self._state = "reach_door_floor_xy"
        self.gripper_action = -1

    def fsm_step(
        self,
        ee_pos,
        ee_quat,
        gripper_width,
        rb_states,
        part_idxs,
        sim_to_april_mat,
        april_to_robot,
        assemble_to,
    ):
        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        body_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        door_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        body_pose = sim_to_april_mat @ body_pose
        door_pose = sim_to_april_mat @ door_pose

        device = ee_pose.device

        if self._state == "reach_door_floor_xy":
            pos = door_pose[:4, 3]

            target_pos = (april_to_robot @ pos)[:3]
            # target_ori = (margin @ april_to_robot @ rot)[:3, :3]
            target_ori = ee_pose[:3, :3]
            target_pos[2] = ee_pos[2]
            # target_pos[1] += 0.01
            target_pos[0] += 0.04
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori),
                ori_noise=torch.tensor([0, 0, 0, 1], device=device),
            )
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02):
                self.prev_pose = target.clone()
                self.prev_pose[2, 3] -= 0.02
                next_state = "reach_door_ori"
        elif self._state == "reach_door_ori":
            target_ori = (april_to_robot @ door_pose)[:3, :3]
            target_pos = self.prev_pose[:3, 3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_door_floor_z"
        elif self._state == "reach_door_floor_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ door_pose)[2, 3]
            target_pos[2] += 0.02
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.007):
                self.prev_pose = target
                next_state = "pick_door"
        elif self._state == "pick_door":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, 0.01, cnt_max=30):
                self.prev_pose = target
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.10], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_near_body"
        elif self._state == "reach_near_body":
            door_pose_robot = april_to_robot @ door_pose
            body_pole_pose_robot = (
                april_to_robot
                @ body_pose
                @ torch.tensor(
                    self.default_assembled_pose,
                    device=device,
                )
                @ torch.tensor(
                    get_mat([0, -0.16, 0], [0, 0, 0]), # Move with 16cm offset.
                    device=device
                )
            )
            # target_door_pose_robot = torch.from_numpy(get_mat([0, np.pi, -np.pi/2], [-0.2, 0, 0])).to(device)
            rel = C.rel_mat(door_pose_robot, body_pole_pose_robot)
            target = rel @ ee_pose
            # Slight up in z position.
            target[2, 3] += 0.007
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "insert_door"
        elif self._state == "insert_door":
            door_pose_robot = april_to_robot @ door_pose
            body_pole_pose_robot = (
                april_to_robot
                @ body_pose
                @ torch.tensor(
                    self.default_assembled_pose,
                    device=device,
                )
            )
            # target_door_pose_robot = torch.from_numpy(get_mat([0, np.pi, -np.pi/2], [-0.2, 0, 0])).to(device)
            rel = C.rel_mat(door_pose_robot, body_pole_pose_robot)
            target = rel @ ee_pose
            # Slight up in z position.
            target[2, 3] += 0.005

            # Clip the target so that it doesn't go too far.
            org_target = target.clone()
            target[:3, 3] = torch.clamp(
                target[:3, 3],
                ee_pos - torch.tensor([0.001, 0.001, 0.001], device=device),
                ee_pos + torch.tensor([0.010, 0.010, 0.010], device=device)
            )
            # Distance in x position.
            if self.satisfy(
                ee_pose,
                org_target,
                max_len=300,
                pos_error_threshold=0.0, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "done"
        if self._state == "done":
            self.gripper_action = -1
            target = self.prev_pose

        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )
