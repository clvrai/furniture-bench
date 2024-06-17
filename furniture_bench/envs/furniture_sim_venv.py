# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Code derived from: https://github.com/isaac-sim/IsaacGymEnvs and https://github.com/transic-robot/transic-envs
"""

try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
except ImportError as e:
    from rich import print

    print(
        """[red][Isaac Gym Import Error]
  1. You need to install Isaac Gym, if not installed.
    - Download Isaac Gym following https://clvrai.github.io/furniture-bench/docs/getting_started/installation_guide_furniture_sim.html#download-isaac-gym
    - Then, pip install -e isaacgym/python
  2. If PyTorch was imported before furniture_bench, please import torch after furniture_bench.[/red]
"""
    )
    print()
    raise ImportError(e)

import numpy as np
import torch
from gym import spaces

import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.sim_config import sim_config
from furniture_bench.config import ROBOT_HEIGHT, config

from furniture_bench.envs.observation import (
    FULL_OBS,
    DEFAULT_VISUAL_OBS,
    DEFAULT_STATE_OBS,
)


class FurnitureSimVEnv(FurnitureSimEnv):
    """Vectorized implementation of original FurnitureSim"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set control limits
        self.cmd_limit = torch.tensor(
            [0.5, 0.5, 0.5, 0.3, 0.3, 0.3], device=self.device
        ).unsqueeze(0)
        self.action_scale = 1.0
        # OSC Gains
        self.kp = torch.tensor(sim_config["robot"]["kp"], device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.tensor([5.0] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

    def step(self, action):
        action = self._process_action(action)
        sim_steps = int(
            1.0
            / config["robot"]["hz"]
            / sim_config["sim_params"].dt
            / sim_config["sim_params"].substeps
            + 0.1
        )
        # Split the arm and the gripper action.
        self.pre_physics_step(action)
        # step physics and render each frame
        for i in range(sim_steps):
            self.isaac_gym.simulate(self.sim)

        if self.camera_obs:
            self.isaac_gym.fetch_results(self.sim, True)
            self.isaac_gym.step_graphics(self.sim)

        obs = self._get_observation()
        self.env_steps += 1

        if self.camera_obs:
            self.isaac_gym.end_access_image_tensors(self.sim)

        # Update viewer
        if not self.headless:
            self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
            self.isaac_gym.sync_frame_time(self.sim)

        return (
            obs,
            self._reward(),
            self._done(),
            {"obs_success": True, "action_success": True},
        )

    def pre_physics_step(self, actions):
        assert actions.shape[-1] == 8
        pos, quat_rot, gripper = actions[:, :3], actions[:, 3:7], actions[:, 7:]
        # rot_angle: (...,)
        # rot_axis: (..., 3)
        rot_angle, rot_axis = C.quat2angle_axis(quat_rot)
        # Match the original FurnitureSim direction.
        rot_axis[:, 1] = -rot_axis[:, 1]
        rot_axis[:, 2] = -rot_axis[:, 2]
        # get rotation along each axis
        rot = torch.stack([rot_angle * rot_axis[..., i] for i in range(3)], dim=-1)
        actions = torch.cat([pos, rot, gripper], dim=-1)
        self.actions = actions.clone().to(self.device)
        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        u_arm = self._compute_osc_torques(dpose=u_arm)
        # Control gripper
        u_fingers = torch.zeros((self.num_envs, 2), device=self.device)

        u_fingers[:, 0] = torch.where(
            u_gripper <= 0.0,
            self.franka_dof_upper_limits[-2].item(),
            self.franka_dof_lower_limits[-2].item(),
        )
        u_fingers[:, 1] = torch.where(
            u_gripper <= 0.0,
            self.franka_dof_upper_limits[-1].item(),
            self.franka_dof_lower_limits[-1].item(),
        )
        pos_action = torch.zeros_like(self.dof_pos)
        torque_action = torch.zeros_like(self.dof_pos)
        grip_action = pos_action[:, 7:9]

        # Write gripper command to appropriate tensor buffer
        grip_action[:, :] = u_fingers
        torque_action[:, :7] = u_arm

        # Deploy actions
        self.isaac_gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(pos_action)
        )
        self.isaac_gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torque_action)
        )

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self.dof_pos[:, :7], self.dof_vel[:, :7]
        mm = self.mm[:, :7, :7]
        mm_inv = torch.inverse(mm)

        m_eef_inv = (
            self.jacobian_eef @ mm_inv @ torch.transpose(self.jacobian_eef, 1, 2)
        )
        m_eef = torch.inverse(m_eef_inv)
        eef_vel = self.eef_state[:, 7:]

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(self.jacobian_eef, 1, 2)
            @ m_eef
            @ (self.kp * dpose - self.kd * eef_vel).unsqueeze(-1)
        )
        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        default_pos = torch.tensor(self.default_dof_pos, device=self.device)
        j_eef_inv = m_eef @ self.jacobian_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (default_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null[:, 7:] *= 0
        u_null = mm @ u_null.unsqueeze(-1)
        u += (
            torch.eye(7, device=self.device).unsqueeze(0)
            - torch.transpose(self.jacobian_eef, 1, 2) @ j_eef_inv
        ) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(
            u.squeeze(-1),
            -self._franka_effort_limits[:7].unsqueeze(0),
            self._franka_effort_limits[:7].unsqueeze(0),
        )

        return u


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


class FurnitureSimFullVEnv(FurnitureSimVEnv):
    def __init__(self, **kwargs):
        super().__init__(obs_keys=FULL_OBS, **kwargs)
