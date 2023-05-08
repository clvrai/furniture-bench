"""Define the types of robot state"""
from dataclasses import dataclass
from enum import Enum

import numpy as np


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
