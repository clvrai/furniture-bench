"""Define the types of robot state"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


# List of robot state we are going to use during training and testing.
ROBOT_STATES = [
    "ee_pos",
    "ee_quat",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
]

ROBOT_STATE_DIMS = {
    "ee_pos": 3,
    "ee_quat": 4,
    "ee_pos_vel": 3,
    "ee_ori_vel": 3,
    "joint_positions": 7,
    "joint_velocities": 7,
    "joint_torques": 7,
    "gripper_width": 1,
}

def filter_and_concat_robot_state(robot_state):
    current_robot_state = []
    using_torch = None  # To keep track of whether we're using torch or numpy

    for rs in ROBOT_STATES:
        state_item = robot_state[rs]
        # Determine if we are using torch or numpy
        if using_torch is None:
            using_torch = isinstance(state_item, torch.Tensor)
        # Handle scalar case for gripper_width or any other single-value state
        if rs == "gripper_width" and (torch.is_tensor(state_item) and state_item.dim() == 0) or (isinstance(state_item, np.ndarray) and state_item.shape == ()):
            state_item = torch.tensor([state_item.item()]) if using_torch else np.array([state_item])
        current_robot_state.append(state_item)
    # Concatenate using the appropriate library
    if using_torch:
        return torch.cat(current_robot_state, dim=-1)
    else:
        return np.concatenate(current_robot_state, axis=-1)


@dataclass
class PandaState:
    """Define state of Panda arm and end-effector."""

    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_pos_vel: np.ndarray
    ee_ori_vel: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    gripper_width: np.ndarray


class PandaError(Enum):
    OLD_GRIPPER_ERROR = 1
    OK = "Successful"
    Gripper = "Panda gripper server stopped."
    Arm = 2
