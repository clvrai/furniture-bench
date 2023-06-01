import numpy as np
import numpy.typing as npt
import torch
from numpy.linalg import inv

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.utils.pose import get_mat, is_similar_rot, rot_mat
from furniture_bench.config import config
from furniture_bench.furniture.parts.part import Part


class TableTop(Part):
    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)

        self.gripper_action = -1
        self.body_grip_width = 0.01

        self.skill_complete_next_states = [
            "push",
            "go_up",
        ]  # Specificy next state after skill is complete.
        self.reset()

    def reset(self):
        self.pre_assemble_done = False
        self._state = "reach_body_grasp_xy"
        self.gripper_action = -1

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False

    def _find_closest_y(self, pose):
        closest_y = pose.clone()
        for i in range(4):
            tmp_pose = pose @ torch.tensor(
                self.rel_pose_from_center[self.tag_ids[i]]
            ).float().to(pose.device)
            if tmp_pose[1, 3] < closest_y[1, 3]:
                closest_y = tmp_pose
        return closest_y

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
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        body_pose = sim_to_april_mat @ body_pose
        device = body_pose.device

        if self._state == "reach_body_grasp_xy":
            body_pose = self._find_closest_y(body_pose)
            rot = body_pose[:4, :4] @ torch.tensor(
                rot_mat([np.pi / 2, 0, 0], hom=True), device=device
            )
            pos = body_pose[:3, 3]
            pos = torch.concat([pos, torch.tensor([1.0], device=device)])

            target_pos = (april_to_robot @ pos)[:3]
            target_ori = (april_to_robot @ rot)[:3, :3]
            target_pos[2] = ee_pos[2]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_body_grasp_z"
        if self._state == "reach_body_grasp_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ body_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori),
                pos_noise=torch.normal(
                    mean=torch.zeros((3,)), std=torch.tensor([0.01, 0.01, 0.001])
                ).to(device),
            )
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                self.gripper_action = 1
                next_state = "pick_body"
        if self._state == "pick_body":
            target = self.prev_pose
            self.gripper_action = 1
            # if gripper_width <= self.body_grip_width + 0.005:
            #     self.prev_pose = target
            #     next_state = "push"
            if self.gripper_less(gripper_width, self.body_grip_width):
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
            target_pos[0] -= self.half_width * 2
            target_pos[1] -= self.half_width
            # Margin
            # target_pos[0] -= 0.02
            # target_pos[1] -= 0.01
            target_pos[2] = ee_pose[2, 3]  # Keep z the same.
            target_pos = target_pos[:3]
            target_ori = self.prev_pose[:3, :3]
            target_ori *= 0
            target_ori[0][1] = 1
            target_ori[1][0] = 1
            target_ori[2][2] = -1

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
