import torch
import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.utils.pose import cosine_sim
from furniture_bench.furniture.parts.leg import Leg
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import config


class RoundTableLeg(Leg):
    def __init__(self, part_config, part_idx):
        self.reset_x_len = 0.0425
        self.reset_y_len = 0.09125

        self.tag_offset = 0.01625
        self.half_width = 0.01625
        super().__init__(part_config, part_idx)

        self.reset_gripper_width = 0.05
        self.grasp_margin_x = 0.020
        self.grasp_margin_z = 0.03
        self.insert_height = 0.054

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        return cosine_sim(reset_ori[:3, 1], pose[:3, 1]) > 0.9

    def reset(self):
        self.prev_pose = None
        self._state = "reach_leg_floor_xy"
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
        def rot_mat_tensor(x, y, z, device):
            return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()

        def rel_rot_mat(s, t):
            s_inv = torch.linalg.inv(s)
            return t @ s_inv

        next_state = self._state

        ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
        base_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        base_pose = sim_to_april_mat @ base_pose
        leg_pose = sim_to_april_mat @ leg_pose

        margin = rot_mat_tensor(0, -np.pi / 5, 0, ee_pose.device)
        device = ee_pose.device

        def find_leg_pose_x_look_front(leg_pose):
            best_leg_pose = leg_pose.clone()
            tmp_leg_pose = leg_pose
            rot = rot_mat_tensor(0, -np.pi / 2, 0, device)
            for i in range(3):
                tmp_leg_pose = tmp_leg_pose @ rot
                if best_leg_pose[0, 0] < tmp_leg_pose[0, 0]:
                    best_leg_pose = tmp_leg_pose
            return best_leg_pose

        if self._state == "reach_leg_floor_xy":
            leg_pose = self._find_down_z(leg_pose).clone().to(device)
            # Margin for leg pose
            leg_pose = (
                torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ leg_pose
            )
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            pos = leg_pose[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            target_ori = ee_pose[:3, :3]
            target_pos[2] = ee_pos[2]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori),
                ori_noise=torch.tensor([0, 0, 0, 1], device=device),
            )
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02):
                self.prev_pose = target.clone()
                self.prev_pose[2, 3] -= 0.02
                next_state = "reach_leg_ori"
        elif self._state == "reach_leg_ori":
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            theta_y = torch.acos(leg_pose[1, 1]).detach().cpu().numpy()
            sign = 1 if leg_pose[0, 1] > 0 else -1
            target_ori = (
                rot_mat_tensor(0, 0, sign * theta_y, device)
                @ margin
                @ april_to_robot
                @ rot
            )[:3, :3]
            # Get the y-axis rotation.
            target_pos = (
                april_to_robot
                @ torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ leg_pose[:4, 3]
            )[:3]
            target_pos[2] = ee_pos[2]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_leg_floor_z"
        elif self._state == "reach_leg_floor_z":
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)
            theta_y = torch.acos(leg_pose[1, 1]).detach().cpu().numpy()
            sign = 1 if leg_pose[0, 1] > 0 else -1
            target_ori = (
                rot_mat_tensor(0, 0, sign * theta_y, device)
                @ margin
                @ april_to_robot
                @ rot
            )[:3, :3]
            # Get the y-axis rotation.
            target_pos = (
                april_to_robot
                @ torch.tensor(get_mat([0, 0.043, 0], [0, 0, 0]), device=device)
                @ leg_pose[:4, 3]
            )[:3]
            target_pos[2] += 0.01  # Margin.
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.007):
                self.prev_pose = target
                next_state = "pick_leg"
        elif self._state == "pick_leg":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.13], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            target_pos = torch.tensor([0.5, 0.10, 0.13], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "match_leg_ori"
        elif self._state == "match_leg_ori":
            # target_ori = (margin @ rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_ori = (rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = torch.tensor([0.57, 0.10, 0.13], device=device)
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_base_xy"
        elif self._state == "reach_base_xy":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            base_hole_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_hole_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, base_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, base_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, self.prev_pose[2, 3]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_hole_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.3,
                max_len=20,
            ):
                self.prev_pose = target
                next_state = "reach_base_z"
        elif self._state == "reach_base_z":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            base_hole_pose_robot = (
                april_to_robot
                @ base_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_ori = torch.tensor(
                get_mat([0, 0, 0], [np.pi / 2, 0, -np.pi / 4]), device=device
            )[:3, :3]
            target_pos = base_hole_pose_robot[:3, 3]
            target_hole_pose_robot = C.to_homogeneous(target_pos, target_ori)

            rel = rel_rot_mat(leg_pose_robot, target_hole_pose_robot)
            target = rel @ ee_pose
            target[2] += 0.03  # Margin.
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.000,
                ori_error_threshold=0.0,
                max_len=30,
            ):
                self.prev_pose = target
                next_state = "insert_wait"
        elif self._state == "insert_wait":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["round_table"] - 0.001,
            ):
                next_state = "insert"
        elif self._state == "insert":
            # Dummy transition state for skill complete.
            target = self.prev_pose
            next_state = "pre_grasp"
        elif self._state == "release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["round_table"] - 0.001,
            ):
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            target_ori = rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            target_pos = (april_to_robot @ leg_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, max_len=30):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, 2 * self.half_width + 0.001):
                self.prev_pose = target
                next_state = "screw"
        elif self._state == "screw":
            target_ori = rot_mat_tensor(np.pi, 0, -np.pi / 2 - np.pi / 36, device)[
                :3, :3
            ]
            target_pos = (ee_pos)[:3]
            target_pos[2] -= 0.005
            target = C.to_homogeneous(target_pos, target_ori)

            if self.satisfy(ee_pose, target, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "release"

        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )

    def state_no_noise(self):
        return self._state in [
            "insert_wait",
            "insert_release",
            "insert",
        ]
