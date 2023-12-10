import numpy as np
import numpy.typing as npt
import torch

from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.furniture.parts.part import Part
import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.config import config


class CabinetTop(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        self.rel_pose_from_center[self.tag_ids[0]] = get_mat([0, 0, -0.0275], [0, 0, 0])
        self.rel_pose_from_center[self.tag_ids[1]] = get_mat(
            [0.05875, 0, 0], [0, -np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[2]] = get_mat(
            [0, 0.0, 0.0275], [0, np.pi, 0]
        )
        self.rel_pose_from_center[self.tag_ids[3]] = get_mat(
            [-0.05875, 0.0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[self.tag_ids[4]] = get_mat(
            [0, -0.01375, 0], [-np.pi / 2, 0, 0]
        )

        self.reset_gripper_width = 0.08

        self.reset_x_len = 0.0565
        self.reset_y_len = 0.09525
        self._state = "reach_body_grasp_xy"
        
        self.skill_complete_next_states = [
            "release_gripper",
            "lift_up_top",
            "insert_body",
            "done",
        ]  # Specificy next state after skill is complete. Screw done is handle in `get_assembly_action`
        
        self.body_grip_width = 0.057
        self.width = 0.1

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        return pose[1, 0] > ori_bound or pose[1, 0] <= -ori_bound
    
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
        body_pose = C.to_homogeneous(
            rb_states[part_idxs["cabinet_body"]][0][:3],
            C.quat2mat(rb_states[part_idxs["cabinet_body"]][0][3:7]),
        )

        body_pose = sim_to_april_mat @ body_pose
        device = body_pose.device

        if self._state == "reach_body_grasp_xy":
            pos = torch.concat([body_pose[:3, 3], torch.tensor([1.0]).float().to(body_pose.device)])

            target_pos = (april_to_robot @ pos)[:3]
            target_ori = (
                april_to_robot @
                # C.rot_mat_tensor(0, np.pi / 2 + np.pi / 8, np.pi / 2, device) @
                torch.tensor(
                    get_mat([0, 0, 0], [0, np.pi / 2 + np.pi / 10, np.pi / 2])).to(device).float() @
                body_pose
            )[:3, :3]
            target_pos[2] = ee_pos[2]
            target_pos[0] -= 0.075 # Move beind from the body.
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "reach_body_grasp_z"
        elif self._state == "reach_body_grasp_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ body_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            target[2] -= 0.005
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "pick_body"
        elif self._state == "pick_body":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, 0.01, cnt_max=25):
                self.prev_pose = target
                next_state = "lift_up"
        elif self._state == "lift_up":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.14], device=device
            )
            org_target_pos = target_pos.clone()
            target_pos[2] = torch.clamp(target_pos[2], target_pos[2], ee_pos[2] + 0.03)
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            org_target = C.to_homogeneous(org_target_pos, target_ori)
            if self.satisfy(
                ee_pose, org_target, pos_error_threshold=0.00, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "release_gripper"
        elif self._state == "release_gripper":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["cabinet"] - 0.001,
                cnt_max=20
            ):
                self.prev_pose = target
                next_state = "move_up"
        elif self._state == "move_up": # Go slightly up to avoid collision with the body.
            self.gripper_action = -1
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.04], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.00, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            z = ee_pose[2, 3]
            target_pos = torch.tensor([0.4, -0.05, z], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_top_floor_xy"
        if self._state == "reach_top_floor_xy":
            self.gripper_action = -1
            self.pre_assemble_done = True
            target = self.prev_pose
        
        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action]).to(ee_pos.device),
            skill_complete
        )
    
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
        top_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        body_pose = sim_to_april_mat @ body_pose
        top_pose = sim_to_april_mat @ top_pose

        device = ee_pose.device

        if self._state == "reach_top_floor_xy":
            target_ori = (april_to_robot @
                torch.tensor(
                    get_mat([0, 0, 0], [0, -np.pi / 2, 0])).to(device).float() @
                top_pose
            )[:3, :3]
            pos = top_pose[:4, 3]
            target_pos = (april_to_robot @ pos)[:3]
            target_pos[2] = ee_pos[2] # Keep the z.
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.02, max_len=100):
                self.prev_pose = target.clone()
                next_state = "reach_top_floor_z"
        elif self._state == "reach_top_floor_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ top_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            target[2, 3] += 0.01
            if self.satisfy(ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3):
                self.prev_pose = target
                next_state = "pick_top"
        elif self._state == "pick_top":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, self.body_grip_width + 0.001, cnt_max=20):
                self.prev_pose = target
                next_state = "lift_up_top"
        elif self._state == "lift_up_top":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.19], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "move_center"
        elif self._state == "move_center":
            target_pos = torch.tensor([0.4, 0.0, self.prev_pose[2, 3]], device=device)
            target_ori = self.prev_pose[:3, :3]
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_body_top_xy"
        elif self._state == "reach_body_top_xy":
            body_screw_pose_robot = (
                april_to_robot
                @ body_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            body_screw_pose_robot[:3, :3] = self.prev_pose[:3, :3]
            target = body_screw_pose_robot
            # Keep the z.
            target[2, 3] = self.prev_pose[2, 3]
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0,
                max_len=50
            ):
                self.prev_pose = target
                next_state = "reach_body_top_z"
        elif self._state == "reach_body_top_z":
            body_screw_pose_robot = (
                april_to_robot
                @ body_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target = self.prev_pose
            target[2, 3] = body_screw_pose_robot[2, 3] + 0.02 # Move up a bit.
            body_screw_pose_robot[:3, :3] = self.prev_pose[:3, :3]
            target[:3, :3] = self.prev_pose[:3, :3] # Keep the same orientation.
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "insert_body"
        elif self._state == "insert_body": # Transition for skill labeling.
            target = self.prev_pose
            self.prev_pose = target
            next_state = "screw_gripper"
        elif self._state == "screw_gripper":
            target_pos = self.prev_pose[:3, 3]
            target_ori = (
                torch.tensor(rot_mat([np.pi, 0, -np.pi / 2])).to(device).float()
            )
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.0, ori_error_threshold=0.0
            ):
                self.prev_pose = target
                next_state = "release_screw_gripper"
        elif self._state == "release_screw_gripper":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["cabinet"] - 0.001,
            ):
                self.prev_pose = target
                next_state = "go_up_screw"
        elif self._state == "go_up_screw":
            target_pos = self.prev_pose[:3, 3] + torch.tensor(
                [0, 0, 0.04], device=device
            )
            target_ori = ee_pose[:3, :3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.01, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "pre_grasp_front"
        elif self._state == "pre_grasp_front":
            # Face EE front.
            target = self.prev_pose
            target[:3, :3] =  torch.tensor(rot_mat([np.pi, 0, 0])).to(device).float()
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "pre_grasp_ori"
        elif self._state == "pre_grasp_ori":
            if top_pose[0, 0] > 0:
                target_pose = (april_to_robot @
                    top_pose @
                    torch.tensor(
                        get_mat([0, 0, 0], [-np.pi / 2, 0, 0])).to(device).float()
                )
            else:
                target_pose = (april_to_robot @
                    top_pose @
                    torch.tensor(
                        get_mat([0, 0, 0], [-np.pi / 2, np.pi, 0])).to(device).float()
                )
            target_ori = target_pose[:3, :3]
            target_pos = self.prev_pose[:3, 3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.01):
                self.prev_pose = target.clone()
                next_state = "pre_grasp_z"
        elif self._state == "pre_grasp_z":
            if top_pose[0, 0] > 0:
                target = (april_to_robot @
                    top_pose @
                    torch.tensor(
                        get_mat([0, 0, 0], [-np.pi / 2, 0, 0])).to(device).float()
                )
            else:
                target = (april_to_robot @
                    top_pose @
                    torch.tensor(
                        get_mat([0, 0, 0], [-np.pi / 2, np.pi, 0])).to(device).float()
                )
            if self.satisfy(ee_pose, target, pos_error_threshold=0.01):
                self.prev_pose = target
                next_state = "screw_grasp"
        elif self._state == "screw_grasp":
            target = self.prev_pose
            self.gripper_action = 1
            if self.gripper_less(gripper_width, self.body_grip_width + 0.001):
                self.prev_pose = target
                next_state = "screw_front"
        elif self._state == "screw_front":
            # Face EE front.
            target = self.prev_pose
            target[:3, :3] =  torch.tensor(rot_mat([np.pi, 0, 0])).to(device).float()
            if self.satisfy(ee_pose, target):
                self.prev_pose = target
                next_state = "screw_gripper"
        skill_complete = self.may_transit_state(next_state)

        return (
            target[:3, 3],
            C.mat2quat(target[:3, :3]),
            torch.tensor([self.gripper_action], device=device),
            skill_complete,
        )
