import math
from typing import Dict, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import furniture_bench.controllers.control_utils as C


def diffik_factory(real_robot=True, *args, **kwargs):
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Differential Inverse Kinematics Controller"""

        def __init__(self):
            """Initialize Differential Inverse Kinematics Controller.

            Args:
            """
            super().__init__()
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.pos_scalar = 4.0 / 2.0
            self.rot_scalar = 9.0 / 4.0

            self.scale_errors = True

        def forward(
            self, state_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            # Get states.
            joint_pos_current = state_dict["joint_positions"]

            # 6x7
            jacobian = state_dict["jacobian_diffik"]

            ee_pos, ee_quat = state_dict["ee_pos"], state_dict["ee_quat"]

            position_error = self.goal_pos - ee_pos
            # quat_error = C.quat_mul(self.goal_ori, C.quat_conjugate(ee_quat))

            ee_mat = C.quat2mat(ee_quat).to(joint_pos_current.device)
            goal_mat = C.quat2mat(self.goal_ori).to(joint_pos_current.device)
            mat_error = torch.matmul(goal_mat, torch.inverse(ee_mat))

            if self.scale_errors:
                # position_error = torch.clamp(position_error, min=-0.01, max=0.01)
                max_pos_value = torch.abs(position_error).max()
                clip_pos_value = 0.01
                if max_pos_value > clip_pos_value:
                    pos_error_scalar = clip_pos_value / max_pos_value
                    position_error = position_error * pos_error_scalar

                # rotvec_error = R.from_quat(quat_error.cpu().numpy()).as_rotvec()
                rotvec_error = R.from_matrix(mat_error.cpu().numpy()).as_rotvec()
                delta_norm = np.linalg.norm(rotvec_error)
                max_rot_radians = 0.04

                # one way to do it, manually
                delta_axis = rotvec_error / delta_norm
                delta_norm_clipped = np.clip(
                    delta_norm, a_min=0.0, a_max=max_rot_radians
                )
                delta_rotvec_scaled = delta_axis * delta_norm_clipped
                delta_mat = R.from_rotvec(delta_rotvec_scaled).as_matrix()  # * -1.0
                mat_error = (
                    torch.from_numpy(delta_mat).float().to(joint_pos_current.device)
                )

                # # another way to do it with slerp
                # delta_quat = R.from_rotvec(delta_rotvec_scaled).as_quat() # * -1.0
                # quat_error = torch.from_numpy(delta_quat).float().to(joint_pos_current.device)
                # norm_ratio = max_rot_radians / delta_norm
                # quat_error = C.quat_slerp(torch.Tensor([0., 0., 0., 1.]).float().to(quat_error_pre.device), quat_error_pre, norm_ratio)

            rot_error = R.from_matrix(mat_error.cpu().numpy())

            # position_error = self.ee_pos_error
            # rot_error = self.ee_rot_error

            ee_delta_axis_angle = (
                torch.from_numpy(rot_error.as_rotvec())
                .float()
                .to(position_error.device)
            )

            dt = 1.0
            ee_pos_vel = position_error * self.pos_scalar / dt
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt

            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)
            joint_vel_desired = torch.linalg.lstsq(
                jacobian, ee_velocity_desired
            ).solution
            joint_pos_desired = joint_pos_current + joint_vel_desired * dt

            return {"joint_positions": joint_pos_desired}

        def set_goal(self, goal_pos, goal_ori):
            self.goal_pos = goal_pos
            self.goal_ori = goal_ori

        def reset(self):
            pass

    return DiffIKController(*args, **kwargs)
