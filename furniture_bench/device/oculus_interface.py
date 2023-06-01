try:
    from oculus_reader.reader import OculusReader
except ImportError:
    raise Exception(
        "Please install oculus_reader following https://github.com/rail-berkeley/oculus_reader"
    )

import numpy as np

import furniture_bench.utils.transform as T
from furniture_bench.utils.pose import rot_mat
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.device.device_interface import DeviceInterface


class OculusInterface(DeviceInterface):
    """Define oculus interface to control franka panda.

    To control the robot, user needs to hold the trigger button (index finger).
    The button on the side (thumb) can toggle gripper state (open and close).
    The button B will reset the environment and robot.
    """

    def __init__(self):
        self.oculus = OculusReader()
        self.reset()

        self.positional_move_only = False

    def reset(self):
        self.prev_handle_press = False
        self.prev_gripper = -1

        self.prev_oculus_pos = None
        self.prev_oculus_quat = None

    def get_pose_and_button(self):
        poses, buttons = self.oculus.get_transformations_and_buttons()

        handle_press = buttons.get("RTr", False)
        gripper_press = buttons.get("RG", False)
        joystick_press = buttons.get("RJ", False)
        success = buttons.get("A", False)
        failed = buttons.get("B", False)
        oculus_pose = poses.get("r", None) if handle_press else None

        if success or failed:
            done = CollectEnum.SUCCESS if success else CollectEnum.FAIL
        elif joystick_press:
            print("Joy stick pressed. Terminating.")
            done = CollectEnum.TERMINATE
        else:
            done = CollectEnum.DONE_FALSE

        return oculus_pose, handle_press, gripper_press, joystick_press, done

    def _get_action(self, oculus_pos, oculus_quat):
        rel_oculus_pos = self.prev_oculus_pos - oculus_pos

        # relative movement speed between VR and robot
        action_pos = rel_oculus_pos

        # swap and flip axes
        action_pos = action_pos[[2, 0, 1]]
        action_pos[2] = -action_pos[2]
        action_pos *= 1.4  # Make action_pos larger so that the delta is not too small.

        rel_oculus_quat = T.quat_multiply(
            oculus_quat, T.quat_inverse(self.prev_oculus_quat)
        )

        action_quat = np.array(list(rel_oculus_quat))  # xyzw

        action_pos = np.clip(
            action_pos, -0.1, 0.1
        )  # Maxiumum movement is 0.1m per step.

        if action_quat[3] < 0.0:
            action_quat *= -1.0

        return action_pos, action_quat

    def get_action(self, use_quat=True):
        assert use_quat, "Oculus only works with quaternion"

        (
            oculus_pose,
            handle_press,
            gripper_press,
            joystick_press,
            done,
        ) = self.get_pose_and_button()

        action_pos = np.array([0.0, 0.0, 0.0])
        action_rot = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw
        gripper = self.prev_gripper

        gripper = 1 if gripper_press else -1
        if handle_press:
            oculus_pos = oculus_pose[:3, 3]
            # swap and flip axes
            oculus_mat = rot_mat([-np.pi / 2, 0, np.pi / 2]) @ oculus_pose[:3, :3]
            oculus_quat = T.mat2quat(oculus_mat)

            if self.prev_handle_press:
                action_pos, action_rot = self._get_action(oculus_pos, oculus_quat)

            self.prev_gripper = gripper
            self.prev_oculus_pos = oculus_pos
            self.prev_oculus_quat = oculus_quat

        self.prev_handle_press = handle_press

        if self.positional_move_only:
            action_rot = np.array([0, 0, 0, 1], dtype=np.float32)

        action = np.hstack([action_pos, action_rot, [gripper]])

        return action, done

    def print_usage(self):
        print("==============Oculus Usage=================")
        print("Button on the front: Keep pressing to move along with the robot")
        print("Button on the thumb: Keep pressing to close the gripper")
        print("Joystick button: Terminate the program")
        print("===============================")

    def close(self):
        print("Closing oculus")
        self.oculus.stop()
