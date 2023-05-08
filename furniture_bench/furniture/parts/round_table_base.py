import numpy as np
import torch

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class RoundTableBase(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.reset_x_len = 0.02
        self.reset_y_len = 0.02

        self.rel_pose_from_center[self.tag_ids[0]] = get_mat(
            [0, -0.045625, 0], [0, 0, 0]
        )
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [-0.0456255, 00, 0], [0, 0, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0.045625, 0], [0, 0, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [0.045625, 0, 0], [0, 0, np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.045625, 0.01], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[5]] = get_mat(
            [-0.045625, 0, 0.01], [0, np.pi, -np.pi / 2]
        )
        self.rel_pose_from_center[self.tag_ids[6]] = get_mat(
            [0, 0.045625, 0.01], [0, np.pi, np.pi]
        )
        self.rel_pose_from_center[self.tag_ids[7]] = get_mat(
            [0.045625, 0, 0.01], [0, np.pi, np.pi / 2]
        )
        self._state = "reach_xy"
        self.half_width = 0.02
        self.grasp_margin_z = 0.006

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[2, 2] < -ori_bound

    def heading_down(self, pose):
        return pose[2, 2] > self.reset_ori_bound

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
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        base_pose = sim_to_april_mat @ base_pose
        leg_pose = sim_to_april_mat @ leg_pose

        # margin = torch.tensor(rot_mat([0, -np.pi / 6, 0], hom=True)).to(ee_pose.device)
        # leg_pose = margin @ leg_pose

        if self._state == "reach_xy":
            rot = base_pose[:4, :4] @ torch.tensor(
                rot_mat([np.pi, 0, -np.pi / 4], hom=True)
            ).float().to(base_pose.device)
            pos = base_pose[:4, 3]

            target_pos = (april_to_robot @ pos)[:3]
            target_ori = (april_to_robot @ rot)[:3, :3]
            target_pos[2] = ee_pos[2]

            target = C.to_homogeneous(target_pos, target_ori)

            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_leg_floor_z"
        elif self._state == "reach_leg_floor_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ leg_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            target[2] += self.grasp_margin_z

            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "pick_base"
        elif self._state == "pick_base":
            target = self.prev_pose
            self.gripper_action = 1
            if gripper_width <= 2 * self.half_width + 0.005:
                self.prev_pose = target
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + torch.tensor([0, 0, 0.10]).to(
                ee_pose.device
            )
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "reach_leg_xy":
            leg_hole_pose = leg_pose
            target_pos = (april_to_robot @ table_hole_pose)[:3, 3]
            target_pos[2] = self.prev_pose[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_table_top_z"
        elif self._state == "move_center":
            target_pos = torch.tensor([0.50, 0, 0.10]).to(ee_pose.device)
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_leg_xy"
        # elif self._state == "match_leg_ori":
        #     target_ori = (margin @ torch.tensor(rot_mat([np.pi, 0, 0], hom=True)).float().to(
        #         ee_pose.device))[:3, :3]
        #     target_pos = self.prev_pose[:3, 3]
        #     target = C.to_homogeneous(target_pos, target_ori)
        #     if self.satisfy(ee_pose, target):
        #         self.prev_pose = target
        #         next_state = "reach_table_top_xy"
        elif self._state == "reach_table_top_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = table_pose[2, 3] + 0.085
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "release"
        elif self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if gripper_width >= 0.08 - 0.001:
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            # rot = leg_pose[:4, :4] @ torch.tensor(rot_mat([np.pi / 2, -np.pi / 2, 0],
            #                                               hom=True)).float().to(leg_pose.device)
            # pos = leg_pose[:4, 3]
            # target_ori = (april_to_robot @ rot)[:3, :3]
            # target = C.to_homogeneous(target_pos, target_ori)
            target_ori = torch.tensor(rot_mat([np.pi, 0, 0])).float().to(ee_pose.device)
            target_pos = (april_to_robot @ leg_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            # target_pos[2] += 0.01
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if gripper_width <= 2 * self.half_width + 0.005:
                self.prev_pose = target
                next_state = "screw"
        elif self._state == "screw":
            target_ori = (
                torch.tensor(rot_mat([np.pi, 0, -np.pi / 2 - np.pi / 36]))
                .float()
                .to(ee_pose.device)
            )
            target_pos = (ee_pos)[:3]
            target_pos[2] -= 0.001
            target = C.to_homogeneous(target_pos, target_ori)
            # rel_ori = (torch.linalg.inv(leg_pose) @ table_pose)[:3, :3]
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "release"
        if next_state != self._state:
            print(f"Changing state from {self._state} to {next_state}")
            self._state = next_state

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action]).to(ee_pos.device),
        )
