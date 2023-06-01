"""Code derived from https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic"""
import math
from typing import Dict, List

import torch

import furniture_bench.controllers.control_utils as C


def osc_factory(real_robot=True, *args, **kwargs):
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class OSCController(base):
        """Operational Space Control"""

        def __init__(
            self,
            kp: torch.Tensor,
            kv: torch.Tensor,
            ee_pos_current: torch.Tensor,
            ee_quat_current: torch.Tensor,
            init_joints: torch.Tensor,
            position_limits: torch.Tensor,
            mass_matrix_offset_val: List[float] = [0.2, 0.2, 0.2],
            max_dx: float = 0.005,
            controller_freq: int = 1000,
            policy_freq: int = 5,
            ramp_ratio: float = 1,
            joint_kp: float = 10.0,
        ):
            """Initialize EE Impedance Controller.

            Args:
                kp (torch.Tensor): positional gain for determining desired torques based upon the pos / ori errors.
                                    Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)
                kv (torch.Tensor): velocity gain for determining desired torques based on vel / ang vel errors.
                                    Can be either a scalar (same value for all action dims) or list (specific values for each dim).
                                    If kv is defined, damping is ignored.
                ee_pos_current (torch.Tensor): Current end-effector position.
                ee_quat_current (torch.Tensor): Current end-effector orientation.
                init_joints (torch.Tensor): Initial joint position (for nullspace).
                position_limits (torch.Tensor): Limits (m) below and above which the magnitude
                                                of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value for all
                                                cartesian dims), or a 2-list of list (specific min/max values for each dim)
                mass_matrix_offset_val (list): 3f list of offsets to add to the mass matrix diagonal's last three elements.
                                                Used for real robots to adjust for high friction at end joints.
                max_dx (float): Maximum delta of positional movement in interpolation.
                control_freq (int): Frequency of control loop.
                policy_freq (int): Frequency at which actions from the robot policy are fed into this controller
                ramp_ratio (float): Ratio of control_freq / policy_freq. Used to determine how many steps to take in the interpolator.
                joint_kp (float): Proportional gain for joint position control.
            """
            super().__init__()
            # limits
            self.position_limits = position_limits
            self.kp = kp
            self.kv = kv
            self.init_joints = init_joints

            self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
            self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)

            # self.mass_matrix = torch.zeros((7, 7))
            self.mass_matrix_offset_val = mass_matrix_offset_val
            self.mass_matrix_offset_idx = torch.tensor([[4, 4], [5, 5], [6, 6]])

            self.repeated_torques_counter = 0
            self.num_repeated_torques = 3
            self.prev_torques = torch.zeros((7,))

            # Interpolator pos, ori
            self.max_dx = max_dx  # Maximum allowed change per interpolator step
            self.total_steps = math.floor(
                ramp_ratio * float(controller_freq) / float(policy_freq)
            )  # Total num steps per interpolator action
            # Save previous goal
            self.goal_pos = ee_pos_current.clone()
            self.prev_goal_pos = ee_pos_current.clone()
            self.step_num_pos = 1

            self.fraction = 0.5
            self.goal_ori = ee_quat_current.clone()
            self.prev_goal_ori = ee_quat_current.clone()
            self.step_num_ori = 1
            self.prev_interp_pos = ee_pos_current.clone()
            self.prev_interp_ori = ee_quat_current.clone()

            self.joint_kp = joint_kp

        def forward(
            self, state_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            self.repeated_torques_counter = (
                self.repeated_torques_counter + 1
            ) % self.num_repeated_torques
            if self.repeated_torques_counter != 1:
                return {"joint_torques": self.prev_torques}
            # Get states.
            joint_pos_current = state_dict["joint_positions"]
            joint_vel_current = state_dict["joint_velocities"]

            mass_matrix = state_dict["mass_matrix"].reshape(7, 7).t()
            mass_matrix[4, 4] += self.mass_matrix_offset_val[0]
            mass_matrix[5, 5] += self.mass_matrix_offset_val[1]
            mass_matrix[6, 6] += self.mass_matrix_offset_val[2]

            ee_pose = state_dict["ee_pose"].reshape(4, 4).t().contiguous()
            ee_pos, ee_quat = C.mat2pose(ee_pose)
            ee_pos = ee_pos.to(ee_pose.device)
            ee_quat = ee_quat.to(ee_pose.device)

            jacobian = state_dict["jacobian"].reshape(7, 6).t().contiguous()

            ee_twist_current = jacobian @ joint_vel_current
            ee_pos_vel = ee_twist_current[:3]
            ee_ori_vel = ee_twist_current[3:]

            goal_pos = C.set_goal_position(self.position_limits, self.ee_pos_desired)
            goal_ori = self.ee_quat_desired
            # Setting goal_pos, goal_ori.
            self.set_goal(goal_pos, goal_ori)
            goal_pos = self.get_interpolated_goal_pos()
            goal_ori = self.get_interpolated_goal_ori()

            goal_ori_mat = C.quat2mat(goal_ori).to(goal_ori.device)
            ee_ori_mat = C.quat2mat(ee_quat).to(ee_quat.device)

            # Default desired velocities and accelerations are zero.
            ori_error = C.orientation_error(goal_ori_mat, ee_ori_mat)

            # Calculate desired force, torque at ee using control law and error.
            position_error = goal_pos - ee_pos
            vel_pos_error = -ee_pos_vel
            desired_force = torch.multiply(
                position_error, self.kp[0:3]
            ) + torch.multiply(vel_pos_error, self.kv[0:3])

            vel_ori_error = -ee_ori_vel
            desired_torque = torch.multiply(ori_error, self.kp[3:]) + torch.multiply(
                vel_ori_error, self.kv[3:]
            )

            # Calculate Operational Space mass matrix.
            lambda_full, nullspace_matrix = C.opspace_matrices(mass_matrix, jacobian)

            desired_wrench = torch.cat([desired_force, desired_torque])
            decoupled_wrench = torch.matmul(lambda_full, desired_wrench)

            # Project torques that acheive goal into task space.
            torques = torch.matmul(jacobian.T, decoupled_wrench) + C.nullspace_torques(
                mass_matrix,
                nullspace_matrix,
                self.init_joints,
                joint_pos_current,
                joint_vel_current,
                joint_kp=self.joint_kp,
            )

            self._torque_offset(ee_pos, goal_pos, torques)
            self.prev_torques = torques

            return {"joint_torques": torques}

        def set_goal(self, goal_pos, goal_ori):
            if (
                not torch.isclose(goal_pos, self.goal_pos).all()
                or not torch.isclose(goal_ori, self.goal_ori).all()
            ):
                self.prev_goal_pos = self.goal_pos.clone()
                self.goal_pos = goal_pos.clone()
                self.step_num_pos = 1

                self.prev_goal_ori = self.goal_ori.clone()
                self.goal_ori = goal_ori.clone()
                self.step_num_ori = 1
            elif (
                self.step_num_pos >= self.total_steps
                or self.step_num_ori >= self.total_steps
            ):
                self.prev_goal_pos = self.prev_interp_pos.clone()
                self.goal_pos = goal_pos.clone()
                self.step_num_pos = 1

                self.prev_goal_ori = self.prev_interp_ori.clone()
                self.goal_ori = goal_ori.clone()
                self.step_num_ori = 1

        def get_interpolated_goal_pos(self) -> torch.Tensor:
            # Calculate the desired next step based on remaining interpolation steps and increment step if necessary
            dx = (self.goal_pos - self.prev_goal_pos) / (self.total_steps)
            # Check if dx is greater than max value; if it is; clamp and notify user
            if torch.any(abs(dx) > self.max_dx):
                dx = torch.clip(dx, -self.max_dx, self.max_dx)

            interp_goal = self.prev_goal_pos + (self.step_num_pos + 1) * dx
            self.step_num_pos += 1
            self.prev_interp_pos = interp_goal
            return interp_goal

        def get_interpolated_goal_ori(self):
            """Get interpolated orientation using slerp."""
            interp_fraction = (self.step_num_ori / self.total_steps) * self.fraction
            interp_goal = C.quat_slerp(
                self.prev_goal_ori, self.goal_ori, fraction=interp_fraction
            )
            self.step_num_ori += 1
            self.prev_interp_ori = interp_goal

            return interp_goal

        def _torque_offset(self, ee_pos, goal_pos, torques):
            """Torque offset to prevent robot from getting stuck when reached too far."""
            if (
                ee_pos[0] >= self.position_limits[0][1]
                and goal_pos[0] - ee_pos[0] <= -self.max_dx
            ):
                torques[1] -= 2.0
                torques[3] -= 2.0

        def reset(self):
            self.repeated_torques_counter = 0

    return OSCController(*args, **kwargs)
