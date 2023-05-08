"""Miscellaneous functions for the robot."""
import numpy as np

# List of robot state we are going to use during training
ROBOT_STATES = [
    "ee_pos",
    "ee_quat",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
]


def concat_robot_state(robot_state, torch=False):
    current_robot_state = []
    for rs in ROBOT_STATES:
        if rs == "gripper_width" and robot_state[rs].shape == ():
            robot_state[rs] = np.array([robot_state[rs]])
        current_robot_state.append(robot_state[rs])
    return np.concatenate(current_robot_state)
