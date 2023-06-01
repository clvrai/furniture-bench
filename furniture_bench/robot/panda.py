import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch

from furniture_bench.utils.pose import is_similar_rot, rot_mat
from furniture_bench.config import config
from furniture_bench.controllers.osc import osc_factory
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.robot.robot_state import PandaState, PandaError
import furniture_bench.utils.transform as T


class Panda:
    """PandaArm with custom methods."""

    def __init__(
        self,
        robot_config,
        max_gripper_width: float = 0.065,
    ):
        """
        Args:
            robot_config: Robot configuration.
            randomness: Randomize the robot initial pose.
        """

        from polymetis import GripperInterface, RobotInterface

        if config["robot"]["server_ip"] == "":
            from rich import print

            print(f"[bold red]SERVER_IP is not defined.[/bold red]")
            raise ValueError(f"SERVER_IP is not defined.")

        self.robot_config = robot_config
        if robot_config["server_ip"] is None:
            raise ValueError("Please specify the server IP address.")

        self.arm = RobotInterface(
            ip_address=robot_config["server_ip"], enforce_version=False
        )
        self.gripper = GripperInterface(ip_address=robot_config["server_ip"])

        self.reset_joints = torch.tensor(robot_config["reset_joints"])

        self.arm.set_home_pose(self.reset_joints)
        self.dof = 7

        # Positinoal and velocity gains for robot control.
        # self.kp = torch.tensor([40, 40, 40, 25.0, 20.0, 25.0])
        self.kp = torch.tensor([80, 80, 80, 50.0, 40.0, 50.0])
        self.kv = torch.ones((6,)) * torch.sqrt(self.kp) * 2.0

        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open an close actions.

        self.max_gripper_width = max_gripper_width

        self.max_go_time = 2.5  # 2.5 seconds maximum to move the robot.

        # Count how many times the robot has stopped moving in a row.
        # This is used to declare "done" when the robot stopped moving.
        self.motion_stopped_counter = 0

    def init_controller(self, kp: torch.Tensor, kv: torch.Tensor):
        """Initialize the OSC controller.

        Args:
            kp: Position gain.
            kv: Velocity gain.
        """
        ee_pos_current, ee_quat_current = self.get_ee_pose()
        ee_pos_current = torch.tensor(ee_pos_current, dtype=torch.float32)
        ee_quat_current = torch.tensor(ee_quat_current, dtype=torch.float32)
        self.ctrl = osc_factory(
            ee_pos_current=ee_pos_current,
            ee_quat_current=ee_quat_current,
            init_joints=self.reset_joints,
            kp=kp,
            kv=kv,
            position_limits=torch.tensor(self.robot_config["position_limits"]),
        )
        self.arm.send_torch_policy(torch_policy=self.ctrl, blocking=False)

    def get_state(self) -> Tuple[Optional[PandaState], PandaError]:
        """Get state of the Panda arm and end-effector."""
        robot_state = self.arm.get_robot_state()
        gripper_state = self.gripper.get_state()
        if gripper_state is None:
            print("Could not get gripper state. Please rerun the gripper server.")
            return None, PandaError.Gripper

        ee_pos, ee_quat = T.mat2pose(np.array(robot_state.ee_pose).reshape(4, 4).T)
        jacobian = torch.tensor(robot_state.jacobian).reshape(7, 6).T

        ee_twist = jacobian @ torch.tensor(robot_state.joint_velocities)
        ee_pos_vel = ee_twist[:3]
        ee_ori_vel = ee_twist[3:]

        return (
            PandaState(
                joint_positions=np.array(robot_state.joint_positions),
                joint_velocities=np.array(robot_state.joint_velocities),
                joint_torques=np.array(robot_state.joint_torques_computed),
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                ee_pos_vel=ee_pos_vel.numpy(),
                ee_ori_vel=ee_ori_vel.numpy(),
                gripper_width=np.array([gripper_state.width], dtype=np.float32),
            ),
            PandaError.OK,
        )

    def execute(
        self, action: npt.NDArray[np.float32], action_filtering: bool = True
    ) -> bool:
        """Execute robot action.

        Args:
            action: Action to execute, 7D for arm, 1D for the gripper.
        Returns: True if the action was successful, False otherwise.
        """
        # Setup frequencly.
        arm_action, grasp = action[:-1], action[-1]

        if np.abs(arm_action[:3]).max() > 0.11:  # 11 cm.
            if action_filtering:
                print(f"[env] Position action too big: {arm_action[:3]}, skipping it.")
                return False
            else:
                # Clip the action to be within the range.
                arm_action[:3] = np.clip(arm_action[:3], -0.10, 0.10)
        # Arm action.
        ee_pos, ee_quat = self.get_ee_pose()
        goal_ee_pos = torch.tensor(ee_pos, dtype=torch.float32) + torch.tensor(
            arm_action[:3], dtype=torch.float32
        )
        act_quat = arm_action[3:]

        goal_ee_quat = torch.tensor(
            T.quat_multiply(ee_quat, act_quat), dtype=torch.float32
        )
        self.arm.update_desired_ee_pose(position=goal_ee_pos, orientation=goal_ee_quat)
        # Gripper action.

        if (
            np.sign(grasp) != np.sign(self.last_grasp)
            and not self.gripper.get_state().is_moving
            and np.abs(grasp) > self.grasp_margin
        ):
            self._change_gripper_state(grasp)
            self.last_grasp = grasp

        if self._robot_stopped():
            self.motion_stopped_counter += 1
        else:
            self.motion_stopped_counter = 0

        return True

    def _motion_stopped_for_too_long(self) -> bool:
        """Check if the robot has stopped for too long."""
        if (
            self.motion_stopped_counter
            > self.robot_config["motion_stopped_counter_threshold"]
        ):
            print("[env] Robot stopped for too long.")
            self.motion_stopped_counter = 0
            return True

        return False

    def _robot_stopped(self) -> bool:
        return self.arm.get_joint_velocities().abs().max() < 0.0055

    def _change_gripper_state(self, grasp: float):
        if grasp < 0:
            self.open_gripper()
        else:
            self.close_gripper()

    def open_gripper(
        self, blocking: bool = False, gripper_width: Optional[float] = None
    ):
        width = self.max_gripper_width if gripper_width is None else gripper_width
        self.gripper.goto(width=width, speed=0.1, force=0.0, blocking=blocking)
        self.last_grasp = -1

    def open_gripper_delta(self, blocking: bool = False):
        width = self.gripper.get_state().width + 0.01
        self.open_gripper(blocking=blocking, gripper_width=width)

    def close_gripper(self, blocking: bool = False):
        self.gripper.grasp(speed=0.1, force=9.0, blocking=blocking)
        self.last_grasp = 1

    def get_ee_pose(self):
        ee_pose = self.arm.get_ee_pose_mat()
        ee_pose = ee_pose.numpy()
        return T.mat2pose(ee_pose)

    def init_reset(self):
        self.open_gripper()
        self.arm.go_home(blocking=True)
        self.init_controller(self.kp, self.kv)
        self.last_grasp = -1
        self.motion_stopped_counter = 0
        self.ctrl.reset()

    def reset(self, randomness=Randomness.LOW):
        self.open_gripper()
        self.arm.go_home(blocking=True)
        self.init_controller(self.kp, self.kv)
        if randomness in [
            Randomness.MEDIUM,
            Randomness.MEDIUM_COLLECT,
            Randomness.HIGH,
            Randomness.HIGH_COLLECT,
        ]:
            pos_noise = np.random.uniform(
                low=-config["robot"]["pos_noise_med"],
                high=config["robot"]["pos_noise_med"],
                size=3,
            )
            quat_noise = T.axisangle2quat(
                [
                    np.radians(
                        np.random.uniform(
                            -config["robot"]["rot_noise_med"],
                            config["robot"]["rot_noise_med"],
                        )
                    ),
                    np.radians(
                        np.random.uniform(
                            -config["robot"]["rot_noise_med"],
                            config["robot"]["rot_noise_med"],
                        )
                    ),
                    np.radians(
                        np.random.uniform(
                            -config["robot"]["rot_noise_med"],
                            config["robot"]["rot_noise_med"],
                        )
                    ),
                ]
            )
            self.go_delta(pos_noise, quat_noise)
        self.last_grasp = -1
        self.motion_stopped_counter = 0

        # TODO: Add failure checking.
        return True

    def gripper_face_front(self):
        self.go_rot_mat(rot_mat([np.pi, 0, 0]))

    def gripper_face_back(self):
        self.go_rot_mat(rot_mat([np.pi, 0, 0]) @ rot_mat([0, 0, np.pi]))

    def update_sleep(
        self,
        position: torch.Tensor,
        orientation: Optional[torch.Tensor] = None,
        sleep_time=2.0,
    ):
        """Update the end-effector pose and sleep for a given amount of time."""
        self.arm.update_desired_ee_pose(position=position, orientation=orientation)
        time.sleep(sleep_time)

    def go(
        self,
        goal_pos: Union[npt.NDArray[np.float32], list],
        goal_quat: Union[npt.NDArray[np.float32], list],
        z_last: bool = True,
    ) -> None:
        """Go to a desired pose within given time limit.

        Args:
            goal_pos: Goal position in robot coordinate.
            goal_quat: Goal orientation in robot coordinate.
            z_last: Whether the z-positional move should be done after every other moves.
        """
        if isinstance(goal_pos, list):
            goal_pos = np.array(goal_pos)
        if goal_pos.shape == (4,):
            # Homogeneous.
            goal_pos = goal_pos[:3]
        if isinstance(goal_quat, list):
            goal_quat = np.array(goal_quat)

        ee_pos, ee_quat = self.get_ee_pose()
        if z_last:
            same_z = goal_pos.copy()
            same_z[2] = ee_pos[2]
            self.go(same_z, goal_quat, z_last=False)

        start = time.time()
        while not (np.abs(ee_pos - goal_pos) < 0.005).all() or not is_similar_rot(
            T.quat2mat(ee_quat), T.quat2mat(goal_quat)
        ):
            self.update_sleep(
                position=torch.tensor(goal_pos),
                orientation=torch.tensor(goal_quat),
                sleep_time=0.2,
            )

            if time.time() - start > self.max_go_time:
                break
            ee_pos, ee_quat = self.get_ee_pose()

    def go_mat(self, goal_mat: npt.NDArray[np.float32]):
        """Matrix form input of `self.go` method."""
        goal_pos, goal_quat = T.mat2pose(goal_mat)
        self.go(goal_pos, goal_quat)

    def go_nearest_90_z(self):
        """Rotate end-effector to the nearest 90 degree angle in the z axis."""
        # Find the nearest 90 degree angle.
        ee_pos, ee_quat = self.get_ee_pose()
        mat = T.quat2mat(ee_quat)
        ee_frame_mat = rot_mat([np.pi, 0, 0]) @ mat
        euler_angles = T.mat2euler(ee_frame_mat)
        ee_z = math.degrees(euler_angles[2])
        sign = np.sign(ee_z)
        ee_z = np.abs(ee_z)

        goal_z = round(ee_z / 90) * 90

        if sign < 0:
            goal_z = -goal_z

        euler_angles[2] = math.radians(goal_z)
        robot_frame_mat = rot_mat([-np.pi, 0, 0]) @ T.euler2mat(euler_angles)
        goal_quat = T.mat2quat(robot_frame_mat)
        self.go(ee_pos, goal_quat)

    def go_pos(self, goal_pos):
        _, ee_quat = self.get_ee_pose()
        self.go(goal_pos, ee_quat, z_last=True)

    def go_rot(self, goal_quat):
        ee_pos, _ = self.get_ee_pose()
        self.go(ee_pos, goal_quat)

    def go_rot_mat(self, rot_mat):
        goal_quat = T.mat2quat(rot_mat)
        self.go_rot(goal_quat)

    def go_delta(self, delta_pos, delta_quat):
        ee_pos, ee_quat = self.get_ee_pose()
        goal_pos = ee_pos + delta_pos
        goal_quat = T.quat_multiply(delta_quat, ee_quat)
        self.go(goal_pos, goal_quat, z_last=False)

    def go_delta_xy(self, delta_xy):
        ee_pos, ee_quat = self.get_ee_pose()
        goal_pos = ee_pos
        goal_pos[0] = ee_pos[0] + delta_xy[0]
        goal_pos[1] = ee_pos[1] + delta_xy[1]
        self.go(goal_pos, ee_quat)

    def go_delta_pos(self, delta_pos):
        if isinstance(delta_pos, list):
            delta_pos = np.array(delta_pos)
        ee_pos, ee_quat = self.get_ee_pose()
        goal_pos = ee_pos + delta_pos
        self.go(goal_pos, ee_quat)

    def go_delta_quat(self, delta_quat):
        ee_pos, ee_quat = self.get_ee_pose()
        goal_quat = T.quat_multiply(ee_quat, delta_quat)
        self.go(ee_pos, goal_quat)

    def rotate_z(self, pi: bool):
        ee_pos, _ = self.get_ee_pose()
        # this is due to ee-frame and robot frame is different.
        # rotate robot_frame x-axis pi is ee_frame.
        rot = rot_mat([np.pi, 0, 0])
        if pi:
            rot = rot_mat([0, 0, np.pi]) @ rot
        rot = T.mat2quat(rot)

        self.go(ee_pos, rot)

    def tilt_place(self, pos):
        self.go_delta_pos([0, 0, 0.1])
        self.move_xy(pos)
        self.tilt()
        self.z_move(pos[2])
        self.open_gripper_delta(blocking=True)

    def tilt(self):
        goal_rot = T.mat2quat(rot_mat([0, np.pi / 2 + np.pi / 6, np.pi]))
        self.go_rot(goal_rot)

    def tilt_ee(self, angles: List[float]):
        """Tilt the end-effector (x,y,z) given angles."""
        assert len(angles) == 3
        goal = self.ee_to_robot_coord(
            rot_mat([angles[0], angles[1], angles[2]], hom=True)
        )
        self.go_rot(T.mat2quat(goal))

    def move_xy(self, pos: List[float]):
        """Move the end-effector (x,y) given position."""
        ee_pos, _ = self.get_ee_pose()
        if isinstance(pos, list):
            pos = np.array(pos)
        self.go_pos(np.concatenate([pos[:2], ee_pos[2:3]]))

    def move_z(self, z: float):
        ee_pos, _ = self.get_ee_pose()
        self.go_pos(np.concatenate([ee_pos[:2], [z]]))

    def z_move(self, z_pos):
        ee_pos, _ = self.get_ee_pose()
        ee_pos[2] = z_pos
        self.go_pos(ee_pos)

    def ee_clock_grasp_mat(self):
        return T.to_homogeneous([0.0, 0.0, 0.0], rot_mat([-np.pi / 2, np.pi, 0]))

    def ee_counter_clock_grasp_mat(self):
        return T.to_homogeneous(
            [
                0.0,
                0.0,
                0.0,
            ],
            rot_mat([-np.pi / 2, 0, 0]),
        )

    def to_robot_coord(self, pose):
        return config["robot"]["tag_base_from_robot_base"] @ pose

    def ee_to_robot_coord(self, pose):
        """Convert end-effector pose to robot coordinate."""
        rot = rot_mat([np.pi, 0, 0], hom=True)
        return rot @ pose

    def check_grasp_success(self):
        gripper_state = self.gripper.get_state()
        if gripper_state.width <= 0.001:
            return False
        return True

    def __del__(self):
        if self.arm is not None:
            self.arm.terminate_current_policy()
