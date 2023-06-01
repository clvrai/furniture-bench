import torch
import numpy as np
import numpy.typing as npt

from furniture_bench.utils.pose import get_mat, is_similar_pos, is_similar_rot, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import config


class CabinetBody(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.name = "cabinet_body"
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, 0.05875], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.0275, 0.0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0, -0.05875], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0.0, 0.0685, 0], [np.pi / 2, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.0685, 0.03], [-np.pi / 2, np.pi / 2, 0]
        )

        self.reset_x_len = 0.1175
        self.reset_y_len = 0.15

        self.reset_gripper_width = 0.06

        self._state = "reach_body_grasp_xy"
        self.pos_error_threshold = 0.015
        self.ori_error_threshold = 0.27
        self.gripper_action = -1
        self.prev_pose = None
        self.pre_assemble_done = False

        self.body_grip_width = 0.01

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
        body_pose = C.to_homogeneous(
            rb_states[part_idxs["cabinet_body"]][0][:3],
            C.quat2mat(rb_states[part_idxs["cabinet_body"]][0][3:7]),
        )

        body_pose = sim_to_april_mat @ body_pose

        if self._state == "reach_body_grasp_xy":
            rot = (
                torch.tensor(rot_mat([0, np.pi / 2, 0], hom=True))
                .float()
                .to(body_pose.device)
                @ body_pose[:4, :4]
            )
            pos = body_pose[:3, 3] + torch.tensor([0.0, 0.07, 0.0]).float().to(
                body_pose.device
            )
            pos = torch.concat([pos, torch.tensor([1.0]).float().to(body_pose.device)])

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
            target_pos[2] = (april_to_robot @ body_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)

            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "pick_body"
        elif self._state == "pick_body":
            target = self.prev_pose
            self.gripper_action = 1
            if gripper_width <= self.body_grip_width + 0.005:
                self.prev_pose = target
                next_state = "push"
        elif self._state == "push":
            target_pos = torch.tensor([0.65, ee_pose[1, 3], ee_pose[2, 3]]).to(
                ee_pose.device
            )
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "release"
        elif self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if gripper_width >= 0.08 - 0.001:
                next_state = "done"
        elif self._state == "done":
            self.gripper_action = -1
            self.pre_assemble_done = True
            target = self.prev_pose
        if next_state != self._state:
            print(f"Changing state from {self._state} to {next_state}")
            self._state = next_state

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action]).to(ee_pos.device),
        )
