import time
from datetime import datetime
from typing import Union
from pathlib import Path

import gym
import cv2
import numpy as np

from furniture_bench.device import make_device
import furniture_bench.utils.transform as T
from furniture_bench.utils.detection import get_cam_to_base
from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.robot.panda import Panda
from furniture_bench.robot.robot_state import PandaState, PandaError
from furniture_bench.utils.draw import draw_axis
from furniture_bench.utils.frequency import set_frequency
from furniture_bench.furniture import furniture_factory
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness, str_to_enum
from furniture_bench.data.collect_enum import CollectEnum


class FurnitureBenchEnv(gym.Env):
    """FurnitureBench base class."""

    def __init__(
        self,
        furniture: str,
        resize_img: bool = True,
        manual_done: bool = False,
        with_display: bool = True,
        draw_marker: bool = False,
        manual_label: bool = False,
        from_skill: int = 0,
        to_skill: int = -1,
        randomness: Union["str", Randomness] = "low",
        high_random_idx: int = -1,
        visualize_init_pose: bool = True,
        record: bool = False,
        manual_reset: bool = True,
    ):
        """
        Args:
            furniture (str): Specifies the type of furniture. Options are 'lamp', 'square_table', 'desk', 'drawer', 'cabinet', 'round_table', 'stool', 'chair', 'one_leg'.
            resize_img (bool): If true, images are resized to 224 x 224.
            manual_done (bool): If true, the episode ends only when the user presses the 'done' button.
            with_display (bool): If true, camera inputs are rendered on environment steps.
            draw_marker (bool): If true and with_display is also true, the AprilTag marker is rendered on display.
            manual_label (bool): If true, manual labeling of the reward is allowed.
            from_skill (int): Skill index to start from (range: [0-5)). Index `i` denotes the completion of ith skill and commencement of the (i + 1)th skill.
            to_skill (int): Skill index to end at (range: [1-5]). Should be larger than `from_skill`. Performs the full task from `from_skill` onwards if not specified (default -1).
            randomness (str): Level of randomness in the environment. Options are 'low', 'med', 'high'.
            high_random_idx (int): Index of the high randomness level (range: [0-2]). Randomly selected if not specified (default -1).
            visualize_init_pose (bool): If true, the initial pose of furniture parts is visualized.
            record (bool): If true, the video of the agent's observation is recorded.
            manual_reset (bool): If true, a manual reset of the environment is allowed.
        """
        super(FurnitureBenchEnv, self).__init__()

        # Check config variables are set.
        for var, name in zip(
            [
                config["camera"][1]["serial"],
                config["camera"][2]["serial"],
                config["camera"][3]["serial"],
            ],
            "CAM_WRIST_SERIAL, CAM_FRONT_SERIAL, CAM_REAR_SERIAL".split(", "),
        ):
            if var == "":
                from rich import print

                print(f"[bold red]{name} is not defined.[/bold red]")
                raise ValueError(f"{name} is not defined.")

        self.furniture_name = furniture
        self.furniture = furniture_factory(furniture)
        self.to_skill = to_skill
        self.from_skill = from_skill
        self.randomness = str_to_enum(randomness)
        self.high_random_idx = high_random_idx
        self.visualize_init_pose = visualize_init_pose
        self.record = record
        self.pose_dim = 3 + 4
        self.resize_img = resize_img
        self.manual_done = manual_done
        self.with_display = with_display
        self.draw_marker = draw_marker
        self.manual_label = manual_label
        self.manual_reset = manual_reset
        if self.manual_reset:
            self.device_interface = make_device("keyboard")

        gym.logger.set_level(gym.logger.INFO)

        if randomness == Randomness.MEDIUM:
            self.furniture.randomize_init_pose(self.from_skill)
        elif randomness == Randomness.HIGH:
            if self.high_random_idx == -1:
                self.high_random_idx = np.random.randint(0, 3)
            self.furniture.randomize_high(self.high_random_idx)

        # Setup a robot.
        max_gripper_width = config["robot"]["max_gripper_width"][self.furniture_name]

        self.robot = Panda(
            robot_config=config["robot"], max_gripper_width=max_gripper_width
        )
        self.robot.init_reset()  # Move to robot to original position.

        # Setup cameras.
        self._get_cam_info()
        self.furniture.start_detection()

        if self.record:
            record_dir = Path(
                f"record_{furniture}/from_skill_{from_skill}_to_skill_{to_skill}"
            )
            record_dir.mkdir(parents=True, exist_ok=True)
            path = record_dir / (datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".avi")
            size = (224 * 2, 224) if self.resize_img else (1280 * 2, 720)
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            self.video_writer = cv2.VideoWriter(str(path), fourcc, 20, size)

    def _get_cam_info(self):
        """Gets intrinsic/extrinsic parameters of Intel RealSense cameras."""
        self.robot.z_move(0.2)  # Move robot up to avoid collision.
        gym.logger.info("Getting camera information...")
        self.cam_intrs = {}
        self.cam_to_base = {}
        for i in range(1, 4):
            cam = RealsenseCam(
                config["camera"][i]["serial"],
                config["camera"]["color_img_size"],
                config["camera"]["depth_img_size"],
                config["camera"]["frame_rate"],
            )
            self.cam_intrs[i] = cam.intr_mat.copy()
            self.cam_to_base[i] = get_cam_to_base(cam, i)
            del cam

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.pose_dim + 1,))

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        dof = self.robot.dof
        img_size = config["furniture"]["env_img_size"]
        robot_state_space = {
            "ee_pos": gym.spaces.Box(low=low, high=high, shape=(3,)),  # (x, y, z)
            "ee_quat": gym.spaces.Box(low=low, high=high, shape=(4,)),  #  (x, y, z, w)
            "ee_pos_vel": gym.spaces.Box(low=low, high=high, shape=(3,)),
            "ee_ori_vel": gym.spaces.Box(low=low, high=high, shape=(3,)),
            "joint_positions": gym.spaces.Box(low=low, high=high, shape=(dof,)),
            "joint_velocities": gym.spaces.Box(low=low, high=high, shape=(dof,)),
            "joint_torques": gym.spaces.Box(low=low, high=high, shape=(dof,)),
            "gripper_width": gym.spaces.Box(low=low, high=high, shape=(1,)),
        }
        return gym.spaces.Dict(
            {
                "robot_state": gym.spaces.Dict(robot_state_space),
                "color_image1": gym.spaces.Box(low=0, high=255, shape=(*img_size, 3)),
                "depth_image1": gym.spaces.Box(low=0, high=255, shape=img_size),
                "color_image2": gym.spaces.Box(low=0, high=255, shape=(*img_size, 3)),
                "depth_image2": gym.spaces.Box(low=0, high=255, shape=img_size),
                "color_image3": gym.spaces.Box(low=0, high=255, shape=(*img_size, 3)),
                "depth_image3": gym.spaces.Box(low=0, high=255, shape=img_size),
                "parts_poses": gym.spaces.Box(
                    low=low,
                    high=high,
                    shape=(self.furniture.num_parts * self.pose_dim,),
                ),
            }
        )

    @set_frequency(config["robot"]["hz"])
    def step(self, action):
        """Robot takes an action.

        Args:
            action:
                np.ndarray of size 8 (dx, dy, dz, x, y, z, w, grip)
        """
        obs, obs_error = self._get_observation()
        action_success = self.robot.execute(action)

        if obs_error != PandaError.OK:
            return None, 0, True, {"obs_success": False, "error": obs_error}

        ret = [
            obs,
            self._reward(),
            self._done(),
            {"action_success": action_success, "obs_success": True},
        ]

        self.env_steps += 1

        if self.manual_reset:
            _, collect_enum = self.device_interface.get_action()
            if collect_enum == CollectEnum.RESET:
                ret[2] = True
        return ret

    def _reward(self):
        """Reward is 1 if two parts are assembled."""
        # If manual_label is True, return 0 since the reward is manually labeled by data_collector.py.
        return 0 if self.manual_label else self.furniture.compute_assemble()

    def _done(self) -> bool:
        if self.manual_done:  # Done will be manually labeled by data_collector.py.
            return False

        if (
            self.furniture.all_assembled()
            or self.furniture.parts_out_pos_lim()
            or self.robot._motion_stopped_for_too_long()
            or self._timeout()
        ):
            gym.logger.info("[env] Finish the episode.")
            return True
        return False

    def _timeout(self) -> bool:
        """Returns True if the episode times out."""
        if self.env_steps > self.furniture.max_env_steps:
            gym.logger.warn("[env] timeout, done")
            return True

        if self.to_skill != -1:
            timeout = (
                self.env_steps > self.furniture.max_env_steps_skills[self.to_skill]
            )
        else:
            timeout = (
                self.env_steps
                > self.furniture.max_env_steps_from_skills[self.from_skill]
            )

        if timeout:
            gym.logger.warn("[env] timeout, done")
        return timeout

    def _get_observation(self):
        """If successful, returns (obs, True); otherwise, returns (None, False)."""
        robot_state, panda_error = self.robot.get_state()
        if panda_error != PandaError.OK:
            return None, panda_error

        (
            parts_poses,
            _,
            color_img1,
            depth_img1,
            color_img2,
            depth_img2,
            color_img3,
            depth_img3,
        ) = self.furniture.get_parts_poses()
        img = cv2.cvtColor(np.hstack([color_img1, color_img2]), cv2.COLOR_RGB2BGR)

        if self.with_display:
            org_img1 = color_img1.copy()
            org_img2 = color_img2.copy()
            org_img3 = color_img3.copy()
            # Make image larger for visibility and show.
            if self.draw_marker:
                img_outs = []
                for img, cam_intr, base in zip(
                    [org_img2, org_img3],
                    [self.cam_intrs[2], self.cam_intrs[3]],
                    [self.cam_to_base[2], self.cam_to_base[3]],
                ):
                    for part_idx in range(self.furniture.num_parts):
                        part_pose = parts_poses[7 * part_idx : 7 * (part_idx + 1)]
                        part_pose = np.linalg.inv(base) @ T.pose2mat(part_pose)
                        if not np.isclose(part_pose[:3], 0).all():
                            img = draw_axis(
                                img,
                                part_pose[:3, :3],
                                part_pose[:3, 3],
                                cam_intr,
                                0.05,
                                5,
                            )
                    base_pose = np.linalg.inv(base) @ np.eye(4)  # Base tag.
                    img = draw_axis(
                        img, base_pose[:3, :3], base_pose[:3, 3], cam_intr, 0.05, 5
                    )
                    img_outs.append(img)

                camera_view = np.hstack(
                    [
                        cv2.cvtColor(img_outs[0], cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(img_outs[1], cv2.COLOR_RGB2BGR),
                    ]
                )
                cv2.imshow("Camera View", camera_view)

            img = np.hstack(
                [
                    resize(cv2.cvtColor(color_img1, cv2.COLOR_RGB2BGR)),
                    resize_crop(cv2.cvtColor(color_img2, cv2.COLOR_RGB2BGR)),
                ]
            )
            cv2.imshow("Resized", img)
            cv2.waitKey(1)
            if self.record:
                self.video_writer.write(img)

        if self.resize_img:
            color_img1 = resize(color_img1)
            depth_img1 = resize(depth_img1)
            color_img2 = resize_crop(color_img2)
            depth_img2 = resize_crop(depth_img2)
            color_img3 = resize(color_img3)
            depth_img3 = resize(depth_img3)

        return (
            dict(
                robot_state=robot_state.__dict__,
                color_image1=color_img1,
                depth_image1=depth_img1,
                color_image2=color_img2,
                depth_image2=depth_img2,
                color_image3=color_img3,
                depth_image3=depth_img3,
                parts_poses=parts_poses,
            ),
            PandaError.OK,
        )

    def _visualize_init_pose(self, draw=True, from_skill=0):
        """Visualizes the pre-defined initial states."""
        part_poses, _, _, _, img2, _, img3, _ = self.furniture.get_parts_poses()

        def _draw(part, draw_img, base, intr):
            reset_pose = np.linalg.inv(base) @ T.to_homogeneous(
                part.reset_pos[from_skill], part.reset_ori[from_skill][:3, :3]
            )
            curr_pose = np.linalg.inv(base) @ T.pose2mat(
                part_poses[part.part_idx * 7 : part.part_idx * 7 + 7]
            )
            draw_img = draw_axis(
                draw_img,
                reset_pose[:3, :3],
                reset_pose[:3, 3],
                intr,
                s=0.05,
                d=6,
                rgb=True,
            )
            draw_img = draw_axis(
                draw_img,
                curr_pose[:3, :3],
                curr_pose[:3, 3],
                intr,
                s=0.05,
                d=6,
                rgb=True,
                trans=True,
            )
            # Draw reset pose.
            rotV, _ = cv2.Rodrigues(reset_pose[:3, :3])
            points = np.float32([0, 0, 0]).reshape(-1, 3)
            projected_points, _ = cv2.projectPoints(
                points, rotV, np.float32(reset_pose[:3, 3]), intr, (0, 0, 0, 0)
            )
            draw_img = cv2.putText(
                draw_img,
                str(part.part_idx),
                org=(
                    projected_points[0, 0, 0].astype(int) + 10,
                    projected_points[0, 0, 1].astype(int) + 10,
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3.5,
                thickness=5,
                color=(255, 0, 0),
            )
            # Draw current pose.
            rotV, _ = cv2.Rodrigues(curr_pose[:3, :3])
            points = np.float32([0, 0, 0]).reshape(-1, 3)
            projected_points, _ = cv2.projectPoints(
                points, rotV, np.float32(curr_pose[:3, 3]), intr, (0, 0, 0, 0)
            )
            draw_img = cv2.putText(
                draw_img,
                str(part.part_idx),
                org=(
                    projected_points[0, 0, 0].astype(int),
                    projected_points[0, 0, 1].astype(int),
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3.5,
                thickness=5,
                color=(51, 153, 255),
            )

            return draw_img

        draw_img2 = img2.copy()
        draw_img3 = img3.copy()
        if draw:
            for part in self.furniture.parts:
                draw_img2 = _draw(
                    part, draw_img2, self.cam_to_base[2], self.cam_intrs[2]
                )
                draw_img3 = _draw(
                    part, draw_img3, self.cam_to_base[3], self.cam_intrs[3]
                )

        img = cv2.cvtColor(np.hstack([draw_img2, draw_img3]), cv2.COLOR_RGB2BGR)
        cv2.imshow("Visualize Initial Pose", img)
        cv2.waitKey(1)
        return img

    def reset(self):
        gym.logger.info("[env] Resetting environment.")

        # Reset robot.
        robot_success = self.robot.reset(self.randomness)
        if not robot_success:
            gym.logger.warn("[env] Cannot reset the robot.")
            return None

        # Reset furniture parts.
        if self.randomness == Randomness.MEDIUM:
            if self.from_skill == 0:
                self.furniture.randomize_init_pose(self.from_skill)
            else:
                self.furniture.randomize_skill_init_pose(self.from_skill)
        elif self.randomness == Randomness.MEDIUM_COLLECT:
            self.furniture.reset_pos_diff_threshold = [0.05, 0.05, 0.05]  # 5cm
            self.furniture.reset_ori_bound = np.cos(np.radians(30))  # 30 degree

        # High randomness only checks if parts can be found.
        check_found_only = self.randomness == Randomness.HIGH_COLLECT
        in_reset_pose = self.furniture.check_parts_in_reset_pose(
            self.from_skill, check_found_only
        )
        self.env_steps = 0
        img = None

        # Check if parts are in reset pose.
        while not in_reset_pose:
            gym.logger.warn("[env] Parts are not in reset pose.")

            if self.visualize_init_pose:
                img = self._visualize_init_pose(
                    not self.randomness == Randomness.HIGH_COLLECT,
                    self.from_skill,
                )
                in_reset_pose = self.furniture.check_parts_in_reset_pose(
                    self.from_skill, check_found_only
                )
            else:
                k = input("Press 'enter' after resetting or 'c' to continue anyway.")
                if k == "":
                    in_reset_pose = self.furniture.check_parts_in_reset_pose(
                        self.from_skill, check_found_only
                    )
                if k == "c":
                    break
            time.sleep(0.5)

        if img is not None:
            draw_img = cv2.putText(
                img,
                "Initialization done",
                org=(50, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                thickness=5,
                color=(53, 153, 255),
            )
            cv2.imshow("Visualize Initial Pose", draw_img)
            cv2.waitKey(1)

        time.sleep(0.1)
        k = input("Press 'enter' to start")
        time.sleep(2.0)

        if self.visualize_init_pose:
            cv2.destroyAllWindows()

        # Move the robot to the initial pose when starting from skill.
        if self.from_skill >= 1:
            if self.randomness == Randomness.MEDIUM:
                pos_noise_abs = config["robot"]["pos_noise_med"]
                rot_noise_abs = config["robot"]["rot_noise_med"]
            else:
                pos_noise_abs = 0.005
                rot_noise_abs = 5

            # Move ee slightly up to avoid collision.
            if (
                self.furniture_name == "drawer"
                and (self.from_skill == 1 or self.from_skill == 2)
                or (self.furniture_name == "desk" and self.from_skill == 4)
            ):
                self.robot.go_delta_pos([0, 0, 0.04])
            # Move ee slightly up to avoid collision.
            if self.furniture_name == "lamp" and (self.from_skill == 4):
                self.robot.go_delta_pos([0, 0, 0.05])

            if self.grasp_and_noise():
                self.grasp(rot_noise_abs)
                if (self.furniture_name == "lamp" and self.from_skill == 3) or (
                    self.furniture_name == "stool"
                    and (self.from_skill == 1 or self.from_skill == 3)
                ):
                    x_noise = np.random.uniform(-pos_noise_abs, 0)
                else:
                    x_noise = np.random.uniform(-pos_noise_abs, pos_noise_abs)
                self.robot.go_delta_xy(
                    [x_noise, np.random.uniform(-pos_noise_abs, pos_noise_abs)]
                )
            else:
                self.robot.move_xy(
                    [
                        self.furniture.furniture_conf["ee_pos"][self.from_skill][0]
                        + np.random.uniform(-pos_noise_abs, pos_noise_abs),
                        self.furniture.furniture_conf["ee_pos"][self.from_skill][1]
                        + np.random.uniform(-pos_noise_abs, pos_noise_abs),
                    ]
                )
                rot = T.quat_multiply(
                    T.axisangle2quat(
                        [
                            np.radians(
                                np.random.uniform(-rot_noise_abs, rot_noise_abs)
                            ),
                            np.radians(
                                np.random.uniform(-rot_noise_abs, rot_noise_abs)
                            ),
                            np.radians(
                                np.random.uniform(-rot_noise_abs, rot_noise_abs)
                            ),
                        ]
                    ),
                    self.furniture.furniture_conf["ee_quat"][self.from_skill],
                )
                self.robot.go_rot(rot)
                # Get how much noise to add to z.
                z_noise = self.furniture.z_noise(self.from_skill)
                if z_noise is None:
                    z_noise = np.random.uniform(-pos_noise_abs, pos_noise_abs)
                self.robot.move_z(
                    self.furniture.furniture_conf["ee_pos"][self.from_skill][2]
                    + z_noise
                )
                if self.furniture.furniture_conf["grippers"][self.from_skill] == 1:
                    self.robot.close_gripper(blocking=True)

        self.furniture.reset()
        obs, obs_error = self._get_observation()
        if obs_error != PandaError.OK:
            gym.logger.warn("[env] Warning: Getting observation was not successful.")
            return None, obs_error
        gym.logger.info("[env] Reset done.")

        return obs

    def grasp_and_noise(self):
        if self.from_skill in [1, 3]:
            if self.furniture_name == "lamp":
                # Make the gripper wider to grasp the lamp bulb as it is quite broad.
                self.robot.open_gripper(blocking=True, gripper_width=0.08)
            return True

        if self.furniture_name == "stool" and self.from_skill == 1:
            self.robot.open_gripper(blocking=True, gripper_width=0.03)
        return False

    def __del__(self):
        if self.record:
            self.video_writer.release()
