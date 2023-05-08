import time
import random
import os
from datetime import datetime
from collections import deque

import gym
import cv2
import numpy as np
from numpy.linalg import inv
from gym import spaces, logger

import furniture_bench.utils.transform as T
from furniture_bench.utils.detection import get_cam_to_base
from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.robot.panda import PandaError
from furniture_bench.robot import Panda
from furniture_bench.utils.draw import draw_axis
from furniture_bench.utils.frequency import set_frequency
from furniture_bench.furniture import furniture_factory
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.device import make_device_interface
from furniture_bench.data.collect_enum import CollectEnum


class FurnitureEnv(gym.Env):
    def __init__(
        self,
        furniture: str,
        use_quat: bool = True,
        resize_img: bool = True,
        manual_done: bool = False,
        with_display: bool = True,
        draw_marker: bool = False,
        manual_label: bool = False,
        from_skill: int = 0,
        to_skill: int = -1,
        randomness: Randomness = Randomness.SKILL_FIXED,
        high_random_idx: int = 0,
        visualize_init_pose: bool = True,
        record_video: bool = False,
        manual_reset: bool = True,
    ):
        """
        Args:
            furniture: The name of the furniture.
            use_quat: Whether to use quaternion or Euler angle for rotation.
            resize_img: Whether to resize the raw camera frame.
            manual_done: Whether to manually finish the episode.
            with_display: Whether to display the environment.
            draw_marker: Whether to draw the marker in display. Only available when with_display=True.
            manual_label: Manually label the rewards. The reward always output 0 when this is True.
            skill: skill to execute (0 is from beginning).
            from_skill: skill to execute from (0 is from beginning).
            randomness: Randomize the environment and the robot initial pose.
            visualize_init_pose: Whether to visualize the initial pose of the furniture.
            record_video: Whether to record the video of agent's observation.
            manual_reset: Whether to manually reset the environment.
        """
        super(FurnitureEnv, self).__init__()

        self.furniture_name = furniture
        self.furniture = furniture_factory(furniture)
        self.to_skill = to_skill
        self.from_skill = from_skill
        self.randomness = randomness
        self.visualize_init_pose = visualize_init_pose
        self.record_video = record_video

        self.manual_reset = manual_reset
        if self.manual_reset:
            self.device_interface = make_device_interface("keyboard")

        self.high_random_idx = high_random_idx

        logger.set_level(logger.INFO)

        if randomness == Randomness.MEDIUM_RANDOM:
            self.furniture.randomize_init_pose(self.from_skill)
        elif randomness == Randomness.HIGH_RANDOM:
            self.furniture.randomize_high(self.high_random_idx)

        max_gripper_width = (
            self.furniture.max_gripper_width
            if self.furniture.max_gripper_width is not None
            else config["robot"]["gripper_max_width"]
        )

        self.robot = Panda(
            robot_config=config["robot"],
            use_quat=use_quat,
            max_gripper_width=max_gripper_width,
        )
        self.robot.init_reset(self.randomness)  # Move to robot to original position.

        self.furniture_name = furniture
        self._get_cam_info()
        self.furniture.start_detection()

        self.use_quat = use_quat

        self.pose_dim = 3 + (4 if self.use_quat else 3)
        self.resize_img = resize_img
        self.manual_done = manual_done
        self.with_display = with_display
        self.draw_marker = draw_marker
        self.manual_label = manual_label
        self.robot_execute_time = []
        self.get_obs_time = []
        self.reward_time = []
        self.env_step = 0

        if self.record_video:
            if not os.path.exists(
                f"record_{furniture}/from_skill_{from_skill}_to_skill_{to_skill}"
            ):
                os.makedirs(
                    f"record_{furniture}/from_skill_{from_skill}_to_skill_{to_skill}"
                )
            path = os.path.join(
                f"record_{furniture}/from_skill_{from_skill}_to_skill_{to_skill}",
                datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".avi",
            )
            # size = (224 * 2, 224) if self.resize_img else (1280 * 2, 720)
            size = (1280 * 2, 720)
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            self.out = cv2.VideoWriter(path, fourcc, 20, size)
        else:
            self.out = None

    def _get_cam_info(self):
        # Add hoc for getting camera intrinsics.
        cam1 = RealsenseCam(
            config["camera"][1]["serial"],
            config["camera"]["color_img_size"],
            config["camera"]["depth_img_size"],
            config["camera"]["frame_rate"],
        )
        cam2 = RealsenseCam(
            config["camera"][2]["serial"],
            config["camera"]["color_img_size"],
            config["camera"]["depth_img_size"],
            config["camera"]["frame_rate"],
        )
        cam3 = RealsenseCam(
            config["camera"][3]["serial"],
            config["camera"]["color_img_size"],
            config["camera"]["depth_img_size"],
            config["camera"]["frame_rate"],
        )
        self.cam_intrs = [
            cam1.intr_mat.copy(),
            cam2.intr_mat.copy(),
            cam3.intr_mat.copy(),
        ]

        self.cam2_to_base = get_cam_to_base(cam2, 2)
        self.cam3_to_base = get_cam_to_base(cam3, 3)
        del cam1
        del cam2
        del cam3

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(self.pose_dim + 1,))

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        return spaces.Dict(
            {
                "robot_state": spaces.Dict(
                    {
                        # For the arm.
                        "ee_pos": spaces.Box(
                            low=low, high=high, shape=(3,)
                        ),  # (x, y, z)
                        "ee_quat": spaces.Box(
                            low=low, high=high, shape=(4,)
                        ),  #  (x, y, z, w)
                        "ee_pos_vel": spaces.Box(low=low, high=high, shape=(3,)),
                        "ee_ori_vel": spaces.Box(low=low, high=high, shape=(3,)),
                        "joint_positions": spaces.Box(
                            low=low, high=high, shape=(self.robot.dof,)
                        ),
                        "joint_velocities": spaces.Box(
                            low=low, high=high, shape=(self.robot.dof,)
                        ),
                        "joint_torques": spaces.Box(
                            low=low, high=high, shape=(self.robot.dof,)
                        ),
                        # For the gripper.
                        "gripper_width": spaces.Box(low=low, high=high, shape=(1,)),
                    }
                ),
                "color_image1": spaces.Box(
                    low=0, high=255, shape=(*config["furniture"]["env_img_size"], 3)
                ),
                "depth_image1": spaces.Box(
                    low=0, high=255, shape=config["furniture"]["env_img_size"]
                ),
                "color_image2": spaces.Box(
                    low=0, high=255, shape=(*config["furniture"]["env_img_size"], 3)
                ),
                "depth_image2": spaces.Box(
                    low=0, high=255, shape=config["furniture"]["env_img_size"]
                ),
                "color_image3": spaces.Box(
                    low=0, high=255, shape=(*config["furniture"]["env_img_size"], 3)
                ),
                "depth_image3": spaces.Box(
                    low=0, high=255, shape=config["furniture"]["env_img_size"]
                ),
                "parts_poses": spaces.Box(
                    low=low, high=high, shape=(self.furniture.num_obj * self.pose_dim,)
                ),
            }
        )

    @set_frequency(config["robot"]["hz"])
    def step(self, action):
        """Robot take action.

        Args:
            action:
                np.ndarray of size 8 (dx, dy, dz, x, y, z, w, grip)    if self.use_quat==True
                np.ndarray of size 7 (dx, dy, dz, dax, day, daz, grip) if self.quat==False
        """
        s = time.time()
        obs, obs_error = self._get_observation()
        self.get_obs_time.append(time.time() - s)
        s = time.time()
        action_success = self.robot.execute(action)
        self.robot_execute_time.append(time.time() - s)

        if obs_error != PandaError.OK:
            return None, 0, True, {"obs_success": False, "error": obs_error}

        s = time.time()
        r = [
            obs,
            self._reward(),
            self._done(),
            {"action_success": action_success, "obs_success": True},
        ]

        self.reward_time.append(time.time() - s)
        self.env_steps += 1

        if self.manual_reset:
            _, collect_enum = self.device_interface.get_action_from_input()
            if collect_enum == CollectEnum.RESET:
                r[2] = True
        return r

    def _reward(self):
        # Reward is the number of assembled parts.
        return self.furniture.compute_assemble() if not self.manual_label else 0

    def _done(self) -> bool:
        if self.manual_done:
            return False
        if (
            self.furniture.all_assembled()
            or self.furniture.parts_out_pos_lim()
            or self.robot._motion_stopped_for_too_long()
            or self.timeout()
        ):
            logger.info("[env] Finish the episode.")
            return True
        return False

    def timeout(self) -> bool:
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
            logger.warn(f"timeout, done")
        return timeout

    def _get_observation(self):
        """If successful, return (obs, True). Otherwise return (None, False)."""
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
                    [self.cam_intrs[1], self.cam_intrs[2]],
                    [self.cam2_to_base, self.cam3_to_base],
                ):
                    for part_idx in range(self.furniture.num_parts):
                        part_pose = parts_poses[7 * part_idx : 7 * (part_idx + 1)]
                        part_pose = np.linalg.inv(base) @ T.pose2mat(part_pose)
                        if not np.isclose(
                            part_pose[:3], np.zeros_like(part_pose[:3])
                        ).all():
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
                    resize_crop(
                        cv2.cvtColor(color_img2, cv2.COLOR_RGB2BGR),
                        config["camera"]["color_img_size"],
                    ),
                ]
            )
            cv2.imshow("Resized", img)
            cv2.waitKey(1)
            if self.record_video:
                self.out.write(img)

        if self.resize_img:
            color_img1 = resize(color_img1)
            depth_img1 = resize(depth_img1)
            color_img2 = resize_crop(color_img2, config["camera"]["color_img_size"])
            depth_img2 = resize_crop(depth_img2, config["camera"]["depth_img_size"])
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
        (
            part_poses,
            _,
            _,
            _,
            color_img2,
            _,
            color_img3,
            _,
        ) = self.furniture.get_parts_poses()

        def _draw(part, draw_img, base, intr):
            reset_pose = inv(base) @ T.to_homogeneous(
                part.reset_pos[from_skill], part.reset_ori[from_skill][:3, :3]
            )
            curr_pose = inv(base) @ T.pose2mat(
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

        draw_img2 = color_img2.copy()
        draw_img3 = color_img3.copy()

        if draw:
            for part in self.furniture.parts:
                draw_img2 = _draw(part, draw_img2, self.cam2_to_base, self.cam_intrs[1])
                draw_img3 = _draw(part, draw_img3, self.cam3_to_base, self.cam_intrs[2])

        img = cv2.cvtColor(np.hstack([draw_img2, draw_img3]), cv2.COLOR_RGB2BGR)
        cv2.imshow("Visualize Initial Pose", img)

        cv2.waitKey(1)
        return img

    def reset(self):
        logger.info("[reset] Resetting environment")
        robot_success = self.robot.reset(self.randomness)
        if not robot_success:
            logger.warn("[reset] Can not reset the robot.")
            return None
        if self.randomness == Randomness.MEDIUM_RANDOM:
            if self.from_skill == 0:
                self.furniture.randomize_init_pose(self.from_skill)
            else:
                self.furniture.randomize_skill_init_pose(self.from_skill)
        elif self.randomness == Randomness.MEDIUM_COLLECT:
            self.furniture.reset_pos_diff_threshold = [0.05, 0.05, 0.05]  # 5cm
            self.furniture.reset_ori_bound = np.cos(np.radians(30))  # 30 degree

        # High randomness only check if parts can be found, regardless of their pose.
        check_found_only = self.randomness == Randomness.HIGH_RANDOM_COLLECT
        in_reset_pose = self.furniture.check_parts_in_reset_pose(
            self.from_skill, check_found_only
        )
        self.env_steps = 0
        img = None

        # Check if parts are in reset pose.
        while not in_reset_pose:
            logger.warn("[env] Parts are not in reset pose.")

            if self.visualize_init_pose:
                img = self._visualize_init_pose(
                    not self.randomness == Randomness.HIGH_RANDOM_COLLECT,
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
            cv2.imwrite("done.png", draw_img)
            cv2.waitKey(1)

        time.sleep(0.1)
        k = input("Press 'enter' to start")
        time.sleep(2.0)

        if self.visualize_init_pose:
            cv2.destroyAllWindows()

        # Move the robot to the initial pose when starting from skill.
        if (
            self.from_skill
            >= 1
            # self.randomness == Randomness.SKILL_FIXED
            # or self.randomness == Randomness.SKILL_RANDOM
        ):
            if self.randomness == Randomness.MEDIUM_RANDOM:
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
            if self.furniture_name == "lamp" and (self.from_skill == 4):
                # Move ee slightly up to avoid collision.
                self.robot.go_delta_pos([0, 0, 0.05])

            if self.grasp_and_noise():
                self.grasp(rot_noise_abs)
                if (self.furniture_name == "lamp" and self.from_skill == 3) or (
                    self.furniture_name == "stool"
                    and (self.from_skill == 1 or self.from_skill == 3)
                ):
                    x_noise = random.uniform(-pos_noise_abs, 0)
                else:
                    x_noise = random.uniform(-pos_noise_abs, pos_noise_abs)
                self.robot.go_delta_xy(
                    [x_noise, random.uniform(-pos_noise_abs, pos_noise_abs)]
                )
            else:
                self.robot.move_xy(
                    [
                        self.furniture.furniture_conf["ee_pos"][self.from_skill][0]
                        + random.uniform(-pos_noise_abs, pos_noise_abs),
                        self.furniture.furniture_conf["ee_pos"][self.from_skill][1]
                        + random.uniform(-pos_noise_abs, pos_noise_abs),
                    ]
                )
                rot = T.quat_multiply(
                    T.axisangle2quat(
                        [
                            np.radians(random.uniform(-rot_noise_abs, rot_noise_abs)),
                            np.radians(random.uniform(-rot_noise_abs, rot_noise_abs)),
                            np.radians(random.uniform(-rot_noise_abs, rot_noise_abs)),
                        ]
                    ),
                    self.furniture.furniture_conf["ee_quat"][self.from_skill],
                )
                self.robot.go_rot(rot)
                # Get how much noise to add to z.
                z_noise = self.furniture.z_noise(self.from_skill)
                if z_noise is None:
                    z_noise = random.uniform(-pos_noise_abs, pos_noise_abs)
                self.robot.move_z(
                    self.furniture.furniture_conf["ee_pos"][self.from_skill][2]
                    + z_noise
                )
                if self.furniture.furniture_conf["grippers"][self.from_skill] == 1:
                    self.robot.close_gripper(blocking=True)

        self.furniture.reset()
        obs, obs_error = self._get_observation()
        if obs_error != PandaError.OK:
            logger.warn("[env] Warning: Getting observation was not successful.")
            return None, obs_error
        logger.info("[env] Reset done")

        return obs

    def grasp_and_noise(self):
        if self.furniture_name in [
            "square_table",
            "desk",
            "round_table",
            "cabinet",
            "drawer",
            "chair",
            "lamp",
            "stool",
        ]:
            if self.from_skill in [1, 3]:
                if self.furniture_name == "lamp":
                    # Make the gripper wider to grasp the lamp bulb as it is quite broad.
                    self.robot.open_gripper(blocking=True, gripper_width=0.08)
                return True

            if self.furniture_name == "stool" and self.from_skill == 1:
                self.robot.open_gripper(blocking=True, gripper_width=0.03)
        return False

    def __del__(self):
        self.out.release()
