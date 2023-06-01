import torch
import numpy as np
import numpy.typing as npt

from furniture_bench.furniture.parts.part import Part
from furniture_bench.utils.pose import get_mat, is_similar_rot, is_similar_xz, rot_mat
from furniture_bench.config import config
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C


class Leg(Part):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)
        tag_ids = part_config["ids"]

        self.rel_pose_from_center[tag_ids[0]] = get_mat(
            [0, 0, -self.tag_offset], [0, 0, 0]
        )
        self.rel_pose_from_center[tag_ids[1]] = get_mat(
            [-self.tag_offset, 0, 0], [0, np.pi / 2, 0]
        )
        self.rel_pose_from_center[tag_ids[2]] = get_mat(
            [0, 0, self.tag_offset], [0, np.pi, 0]
        )
        self.rel_pose_from_center[tag_ids[3]] = get_mat(
            [self.tag_offset, 0, 0], [0, -np.pi / 2, 0]
        )

        self.done = False
        self.pos_error_threshold = 0.01
        self.ori_error_threshold = 0.25

        self.skill_complete_next_states = [
            "lift_up",
            "insert",
        ]  # Specificy next state after skill is complete. Screw done is handle in `get_assembly_action`

        self.reset()

        self.part_attached_skill_idx = 4

    def reset(self):
        self.prev_pose = None
        self._state = "reach_leg_floor_xy"
        self.gripper_action = -1

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

    def _find_down_z(self, mat):
        for _ in range(4):
            if mat[2, 2] > 0.8:  # Z is down.
                break
            mat = mat @ torch.tensor(
                T.rotmat2hom(rot_mat([0, np.pi / 2, 0]))
            ).float().to(mat.device)
        return mat

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
        table_pose = C.to_homogeneous(
            rb_states[part_idxs[assemble_to]][0][:3],
            C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
        )
        leg_pose = C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )

        table_pose = sim_to_april_mat @ table_pose
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
            rot = rot_mat_tensor(np.pi / 2, -np.pi / 2, 0, device)

            pos = leg_pose[:4, 3]

            target_pos = (april_to_robot @ pos)[:3]
            # target_ori = (margin @ april_to_robot @ rot)[:3, :3]
            target_ori = ee_pose[:3, :3]
            target_pos[2] = ee_pos[2]
            target_pos[1] += 0.01
            target_pos[0] += self.grasp_margin_x
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
            target_ori = (margin @ april_to_robot @ rot)[:3, :3]
            target_pos = self.prev_pose[:3, 3]
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target, pos_error_threshold=0.015):
                self.prev_pose = target
                next_state = "reach_leg_floor_z"
        elif self._state == "reach_leg_floor_z":
            target_pos = self.prev_pose[:3, 3]
            target_pos[2] = (april_to_robot @ leg_pose)[2, 3]
            target_ori = self.prev_pose[:3, :3]
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
                next_state = "move_center"
        elif self._state == "move_center":
            target_pos = torch.tensor([0.5, 0.1, 0.1], device=device)
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
            target_ori = (margin @ rot_mat_tensor(np.pi, 0, 0, device))[:3, :3]
            target_pos = torch.tensor([0.57, 0.1, 0.12], device=device)
            target = self.add_noise_first_target(
                C.to_homogeneous(target_pos, target_ori)
            )
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.02, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_table_top_xy"
        elif self._state == "reach_table_top_xy":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, self.prev_pose[2, 3]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.015, ori_error_threshold=0.3
            ):
                self.prev_pose = target
                next_state = "reach_table_top_z"
        elif self._state == "reach_table_top_z":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.09],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose, target, pos_error_threshold=0.007, ori_error_threshold=0.15
            ):
                self.prev_pose = target
                next_state = "insert_wait"
        elif self._state == "insert_wait":
            leg_pose_robot = april_to_robot @ leg_pose
            leg_pose_robot = find_leg_pose_x_look_front(leg_pose_robot)
            table_hole_pose_robot = (
                april_to_robot
                @ table_pose
                @ torch.tensor(
                    get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
                    device=device,
                )
            )
            target_leg_pose_robot = torch.tensor(
                [
                    [1.0, 0.0, 0.0, table_hole_pose_robot[0, 3]],
                    [0.0, 0.0, -1.0, table_hole_pose_robot[1, 3]],
                    [0.0, 1.0, 0.0, table_pose[2, 3] + 0.084],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            )
            rel = rel_rot_mat(leg_pose_robot, target_leg_pose_robot)
            target = rel @ ee_pose
            if self.satisfy(
                ee_pose,
                target,
                pos_error_threshold=0.0,
                ori_error_threshold=0.0,
                max_len=10,
            ):
                self.prev_pose = target
                next_state = "insert_release"
        elif self._state == "insert_release":
            target = self.prev_pose
            self.gripper_action = -1
            if self.gripper_greater(
                gripper_width,
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
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
                config["robot"]["max_gripper_width"]["square_table"] - 0.001,
            ):
                next_state = "pre_grasp"
        elif self._state == "pre_grasp":
            target_ori = rot_mat_tensor(np.pi, 0, 0, device)[:3, :3]
            target_pos = (april_to_robot @ leg_pose[:4, 3])[:3]
            target_pos[2] += self.grasp_margin_z
            # target = self.add_noise_first_target(C.to_homogeneous(target_pos, target_ori))
            target = C.to_homogeneous(target_pos, target_ori)
            if self.satisfy(ee_pose, target):
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
            # 'screw_grasp', 'screw', 'match_leg_ori', 'reach_table_top_xy', 'reach_table_top_z'
            "insert_wait",
            "insert_release",
            "insert",
        ]

    def _find_closest_y(self, pose):
        closest_y = pose.clone()
        for i in range(4):
            tmp_pose = pose @ torch.tensor(
                self.rel_pose_from_center[self.tag_ids[i]]
            ).float().to(pose.device)
            if tmp_pose[1, 3] < closest_y[1, 3]:
                closest_y = tmp_pose
        return closest_y
