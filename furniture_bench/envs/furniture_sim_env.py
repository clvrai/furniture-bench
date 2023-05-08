import isaacgym
import random
import os
import time
import math
from datetime import datetime

import cv2
import numpy as np
import gym
from gym import spaces
from numpy.linalg import inv
from gym import logger
from isaacgym import gymapi, gymtorch

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.controllers.osc import OSCController
from furniture_bench.robot.robot_state import PandaState
from furniture_bench.furniture import furniture_factory
from furniture_bench.sim_config import sim_config
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.envs.initialization_mode import str_to_enum

import torch
import torchvision


ASSET_ROOT = "assets"


class FurnitureSimEnv(gym.Env):
    def __init__(
        self,
        furniture: str,
        num_envs: int = 1,
        use_quat: bool = True,
        resize_img: bool = True,
        manual_done: bool = False,
        headless: bool = False,
        manual_label: bool = False,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        domain_randomization: bool = False,
        init_assembled: bool = False,
        np_step_out: bool = False,
        record: bool = False,
        channel_first: bool = False,
        from_skill: int = 0,
        randomness: Randomness = Randomness.LOW_RANDOM,
        high_random_idx: int = 0,
        save_camera_input: bool = False,
        **kwargs,
    ):
        """
        Args:
            furniture: The name of the furniture.
            num_envs: Number of environments.
            asset_root: The root directory of the assets.
            use_quat: Whether to use quaternion or Euler angle for rotation.
            resize_img: Whether to resize the raw camera frame.
            manual_done: Whether to manually reset the environment.
            headless: Whether to run the environment in headless mode.
            manual_label: Manually label the rewards.
            domain_randomization: Whether to use domain randomization.
            init_assembled: Whether the furniture is assembled in initial states.
            from_skill: The initial skill for execution (0 is from beginning).
            save_camera_input: Save camera inputs at the beginning of the episode.
        """
        super(FurnitureSimEnv, self).__init__()
        self.device = torch.device("cuda", compute_device_id)

        self.furniture = furniture_factory(furniture)
        self.assemble_idx = 0
        # Furniture for each environment (reward, reset).
        self.furnitures = [furniture_factory(furniture) for _ in range(num_envs)]

        self.furniture_name = furniture
        self.num_envs = num_envs
        self.use_quat = use_quat
        self.pose_dim = 3 + (4 if self.use_quat else 3)
        self.resize_img = resize_img
        self.manual_done = manual_done
        self.headless = headless
        self.move_neutral = False
        self.manual_label = manual_label
        self.ctrl_started = False
        self.domain_randomization = domain_randomization
        self.init_assembled = init_assembled
        self.np_step_out = np_step_out
        self.record = record
        self.channel_first = channel_first
        self.from_skill = from_skill
        self.randomness = (
            randomness
            if isinstance(randomness, Randomness)
            else str_to_enum(randomness)
        )
        self.high_random_idx = high_random_idx
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open an close actions.

        self.save_camera_input = save_camera_input

        if self.record:
            size = (
                config["camera"]["color_img_size"][:2]
                if not self.resize_img
                else (224, 224)
            )  # TODO: Change to config.
            path = (
                datetime.now().strftime(f"assembly_{furniture}_%Y-%m-%d-%H:%M:%S")
                + ".mp4"
            )
            self.out = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"MP4V"), 10, size, True
            )
        # Simulator setup.
        self.isaac_gym = gymapi.acquire_gym()
        self.sim = self.isaac_gym.create_sim(
            compute_device_id,
            graphics_device_id,
            gymapi.SimType.SIM_PHYSX,
            sim_config["sim_params"],
        )
        self._create_ground_plane()
        self._setup_light()
        self.import_assets()
        self.create_envs()
        self.set_viewer()
        self.set_camera()
        self.acquire_base_tensors()

        self.isaac_gym.prepare_sim(self.sim)
        self.refresh()

        self.isaac_gym.refresh_actor_root_state_tensor(self.sim)

        self.init_ee_pos, self.init_ee_quat = self.get_ee_pose()

        logger.set_level(logger.INFO)

        self.ctrl_times = []
        self.osc_times = []

    def _create_ground_plane(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.isaac_gym.add_ground(self.sim, plane_params)

    def _setup_light(self):
        l_color = gymapi.Vec3(0.8, 0.8, 0.8)
        l_ambient = gymapi.Vec3(0.35, 0.35, 0.35)
        l_direction = gymapi.Vec3(0.1, -0.03, 0.2)
        self.isaac_gym.set_light_parameters(
            self.sim, 0, l_color, l_ambient, l_direction
        )

    def create_envs(self):
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.franka_pose = gymapi.Transform()

        table_half_width = 0.015
        table_surface_z = table_pos.z + table_half_width
        self.franka_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]

        franka_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_ee_index = franka_link_dict["k_ee_link"]
        self.franka_base_index = franka_link_dict["panda_link0"]
        # Parts assets.
        # Create assets.
        self.part_assets = {}
        for part in self.furniture.parts:
            asset_option = sim_config["asset"][part.name]
            self.part_assets[part.name] = self.isaac_gym.load_asset(
                self.sim, ASSET_ROOT, part.asset_file, asset_option
            )
        # Create envs.
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.env_steps = torch.tensor(
            [
                0,
            ]
            * self.num_envs,
            dtype=torch.int,
            device=self.device,
        )

        self.handles = {}
        self.ee_idxs = []
        self.ee_handles = []
        self.osc_ctrls = []

        self.base_idxs = []
        self.part_idxs = {}
        self.franka_handles = []
        for i in range(self.num_envs):
            env = self.isaac_gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # Add workspace (table).
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)

            table_handle = self.isaac_gym.create_actor(
                env, self.table_asset, table_pose, "table", i, 0
            )
            table_props = self.isaac_gym.get_actor_rigid_shape_properties(
                env, table_handle
            )
            table_props[0].friction = sim_config['table']['friction']
            self.isaac_gym.set_actor_rigid_shape_properties(
                env, table_handle, table_props
            )

            self.base_tag_pose = gymapi.Transform()
            base_tag_pos = T.pos_from_mat(config["robot"]["tag_base_from_robot_base"])
            self.base_tag_pose.p = self.franka_pose.p + gymapi.Vec3(
                base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
            )
            self.base_tag_pose.p.z = table_surface_z
            base_tag_handle = self.isaac_gym.create_actor(
                env, self.base_tag_asset, self.base_tag_pose, "base_tag", i, 0
            )
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)
            bg_handle = self.isaac_gym.create_actor(
                env, self.background_asset, bg_pose, "background", i, 0
            )
            # TODO: Make config
            obstacle_pose = gymapi.Transform()
            obstacle_pose.p = gymapi.Vec3(
                self.base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015
            )
            obstacle_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), 0.5 * math.pi
            )

            obstacle_handle = self.isaac_gym.create_actor(
                env, self.obstacle_front_asset, obstacle_pose, f"obstacle_front", i, 0
            )
            part_idx = self.isaac_gym.get_actor_rigid_body_index(
                env, obstacle_handle, 0, gymapi.DOMAIN_SIM
            )
            if self.part_idxs.get("obstacle_front") is None:
                self.part_idxs["obstacle_front"] = [part_idx]
            else:
                self.part_idxs[f"obstacle_front"].append(part_idx)

            for j, name in enumerate(["obstacle_right", "obstacle_left"]):
                y = -0.175 if j == 0 else 0.175
                obstacle_pose = gymapi.Transform()
                obstacle_pose.p = gymapi.Vec3(
                    self.base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
                    y,
                    table_surface_z + 0.015,
                )
                obstacle_pose.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 0, 1), 0.5 * math.pi
                )

                obstacle_handle = self.isaac_gym.create_actor(
                    env, self.obstacle_side_asset, obstacle_pose, name, i, 0
                )
                part_idx = self.isaac_gym.get_actor_rigid_body_index(
                    env, obstacle_handle, 0, gymapi.DOMAIN_SIM
                )
                if self.part_idxs.get(name) is None:
                    self.part_idxs[name] = [part_idx]
                else:
                    self.part_idxs[name].append(part_idx)
            # Add robot.
            franka_handle = self.isaac_gym.create_actor(
                env, self.franka_asset, self.franka_pose, "franka", i, 0
            )
            self.franka_num_dofs = self.isaac_gym.get_actor_dof_count(
                env, franka_handle
            )

            self.isaac_gym.enable_actor_dof_force_sensors(env, franka_handle)
            self.franka_handles.append(franka_handle)
            franka_dof_props = self.isaac_gym.get_actor_dof_properties(
                env, franka_handle
            )
            franka_dof_props["friction"][:7] = 55.0
            franka_dof_props["friction"][7:] = 70.0  # Gripper.
            self.isaac_gym.set_actor_dof_properties(
                env, franka_handle, franka_dof_props
            )
            # Get global index of hand and base.
            self.ee_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_ee_index, gymapi.DOMAIN_SIM
                )
            )
            self.ee_handles.append(
                self.isaac_gym.find_actor_rigid_body_handle(
                    env, franka_handle, "k_ee_link"
                )
            )
            self.base_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_base_index, gymapi.DOMAIN_SIM
                )
            )
            # Set dof properties.
            franka_dof_props = self.isaac_gym.get_asset_dof_properties(
                self.franka_asset
            )
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(0.0)
            # Grippers
            franka_dof_props["driveMode"][7:].fill(
                gymapi.DOF_MODE_POS
            )  # TODO: Give force. Change to torque control maybe.
            franka_dof_props["stiffness"][7:].fill(200.0)
            franka_dof_props["damping"][7:].fill(60.0)

            self.isaac_gym.set_actor_dof_properties(
                env, franka_handle, franka_dof_props
            )
            # Set initial dof states
            franka_num_dofs = self.isaac_gym.get_asset_dof_count(self.franka_asset)
            self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = np.array(
                config["robot"]["reset_joints"], dtype=np.float32
            )
            self.default_dof_pos[7:] = config["robot"]["gripper_max_width"] / 2
            default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"] = self.default_dof_pos
            self.isaac_gym.set_actor_dof_states(
                env, franka_handle, default_dof_state, gymapi.STATE_ALL
            )
            # Add furniture parts.
            poses = []

            for part in self.furniture.parts:
                pos, ori = self._get_reset_pose(part)
                part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
                part_pose = gymapi.Transform()
                part_pose.p = gymapi.Vec3(
                    part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
                )
                reset_ori = self.april_coord_to_sim_coord(ori)
                part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
                poses.append(part_pose)
                part_handle = self.isaac_gym.create_actor(
                    env, self.part_assets[part.name], part_pose, part.name, i, 0
                )
                self.handles[part.name] = part_handle

                part_idx = self.isaac_gym.get_actor_rigid_body_index(
                    env, part_handle, 0, gymapi.DOMAIN_SIM
                )
                # Set properties of part.
                part_props = self.isaac_gym.get_actor_rigid_shape_properties(
                    env, part_handle
                )
                part_props[0].friction = sim_config['parts']['friction']
                self.isaac_gym.set_actor_rigid_shape_properties(
                    env, part_handle, part_props
                )

                if self.part_idxs.get(part.name) is None:
                    self.part_idxs[part.name] = [part_idx]
                else:
                    self.part_idxs[part.name].append(part_idx)

                if self.domain_randomization:
                    color = gymapi.Vec3(
                        np.random.uniform(0, 1),
                        np.random.uniform(0, 1),
                        np.random.uniform(0, 1),
                    )
                    self.isaac_gym.set_rigid_body_color(
                        env, part_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
                    )
                    self.isaac_gym.set_rigid_body_texture(
                        env,
                        part_handle,
                        0,
                        gymapi.MESH_VISUAL,
                        self.loaded_texture_handle_list[
                            random.randint(0, len(self.loaded_texture_handle_list) - 1)
                        ],
                    )

            self.parts_handles = {}
            for part in self.furniture.parts:
                self.parts_handles[part.name] = self.isaac_gym.find_actor_index(
                    env, part.name, gymapi.DOMAIN_ENV
                )

    def _get_reset_pose(self, part):
        if self.init_assembled:
            if part.name == "chair_seat":
                # Special case handling for chair seat since the assembly of chair back is not available from initialized pose.
                part.reset_pos = [[0, 0.16, -0.035]]
                part.reset_ori = [rot_mat([np.pi, 0, 0], hom=True)]
            attached_part = False
            attach_to = None
            for assemble_pair in self.furniture.should_be_assembled:
                if part.part_idx == assemble_pair[1]:
                    attached_part = True
                    attach_to = self.furniture.parts[assemble_pair[0]]
                    break
            if attached_part:
                attach_part_pos = self.furniture.parts[attach_to.part_idx].reset_pos[0]
                attach_part_ori = self.furniture.parts[attach_to.part_idx].reset_ori[0]
                attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                if part.default_assembled_pose is not None:
                    pose = attach_part_pose @ part.default_assembled_pose
                    pos = pose[:3, 3]
                    ori = T.to_hom_ori(pose[:3, :3])
                else:
                    pos = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0][:4, 3]
                    )
                    pos = pos[:3]
                    ori = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0]
                    )
                part.reset_pos[0] = pos
                part.reset_ori[0] = ori
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        else:
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        return pos, ori

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        if self.headless == False:
            self.viewer = self.isaac_gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            # Point camera at middle env.
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.62)
            middle_env = self.envs[0]
            self.isaac_gym.viewer_camera_look_at(
                self.viewer, middle_env, cam_pos, cam_target
            )

    def set_camera(self):
        self.camera_handles = []
        self.camera_color1 = []
        self.camera_color2 = []
        self.camera_color3 = []
        self.camera_depth1 = []
        self.camera_depth2 = []
        self.camera_depth3 = []
        for i, env in enumerate(self.envs):
            # Handle resize image.

            self.camera_handles.append([])
            camera_properties = gymapi.CameraProperties()
            camera_properties.enable_tensors = True
            if not self.resize_img:
                camera_properties.horizontal_fov = 69.4
                camera_properties.width = sim_config["camera"]["color_img_size"][0]
                camera_properties.height = sim_config["camera"]["color_img_size"][1]
            else:
                camera_properties.horizontal_fov = 40.0
                camera_properties.width = sim_config["camera"]["resized_img_size"][0]
                camera_properties.height = sim_config["camera"]["resized_img_size"][1]

            # Wrist camera.
            if not self.resize_img:
                h1 = self.isaac_gym.create_camera_sensor(
                    self.envs[i], camera_properties
                )
            else:
                wrist_properties = gymapi.CameraProperties()
                wrist_properties.enable_tensors = True
                wrist_properties.horizontal_fov = (
                    55.0  # A special case for wrist camera to have a wider view.
                )
                wrist_properties.width = sim_config["camera"]["resized_img_size"][0]
                wrist_properties.height = sim_config["camera"]["resized_img_size"][1]
                h1 = self.isaac_gym.create_camera_sensor(self.envs[i], wrist_properties)

            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(-0.04, 0, -0.05)
            local_transform.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), np.radians(-70.0)
            )
            self.isaac_gym.attach_camera_to_body(
                h1,
                self.envs[i],
                self.ee_handles[i],
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            self.camera_handles[i].append(h1)

            # Front camera.
            h2 = self.isaac_gym.create_camera_sensor(self.envs[i], camera_properties)
            cam_pos = gymapi.Vec3(0.90, -0.00, 0.65)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            self.isaac_gym.set_camera_location(h2, self.envs[i], cam_pos, cam_target)
            self.camera_handles[i].append(h2)

            # Rear camera.
            h3 = self.isaac_gym.create_camera_sensor(self.envs[i], camera_properties)
            transform = gymapi.Transform()

            transform.p = gymapi.Vec3(
                self.franka_pose.p.x + 0.08, 0, self.franka_pose.p.z + 0.2
            )
            transform.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), np.radians(35.0)
            )
            self.isaac_gym.set_camera_transform(h3, env, transform)
            self.camera_handles[i].append(h3)

            # TODO clean code.
            # Color
            self.camera_color1.append(
                gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], h1, gymapi.IMAGE_COLOR
                    )
                )
            )
            self.camera_color2.append(
                gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], h2, gymapi.IMAGE_COLOR
                    )
                )
            )
            self.camera_color3.append(
                gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], h3, gymapi.IMAGE_COLOR
                    )
                )
            )
            # Depth
            self.camera_depth1.append(
                gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], h1, gymapi.IMAGE_DEPTH
                    )
                )
            )
            self.camera_depth2.append(
                gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], h2, gymapi.IMAGE_DEPTH
                    )
                )
            )
            self.camera_depth3.append(
                gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], h3, gymapi.IMAGE_DEPTH
                    )
                )
            )

    def import_assets(self):
        self.base_tag_asset = self._import_base_tag_asset()
        self.background_asset = self._import_background_asset()
        self.table_asset = self._import_table_asset()
        self.obstacle_front_asset = self._import_obstacle_front_asset()
        self.obstacle_side_asset = self._import_obstacle_side_asset()
        self.franka_asset = self._import_franka_asset()

    def acquire_base_tensors(self):
        # Get rigid body state tensor
        _rb_states = self.isaac_gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _root_tensor = self.isaac_gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_pos = self.root_tensor.view(self.num_envs, -1, 13)[..., 0:3]
        self.root_quat = self.root_tensor.view(self.num_envs, -1, 13)[..., 3:7]

        _forces = self.isaac_gym.acquire_dof_force_tensor(self.sim)
        _forces = gymtorch.wrap_tensor(_forces)
        self.forces = _forces.view(self.num_envs, 9)

        # import torchcontrol as toco

        # self.robot_model = toco.models.RobotModelPinocchio(
        #     urdf_filename=os.path.join(ASSET_ROOT, self.franka_asset_file),
        #     ee_link_name="k_ee_link",
        # )

        # Get DoF tensor
        _dof_states = self.isaac_gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(
            _dof_states
        )  # (num_dofs, 2), 2 for pos and vel.
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 9)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 9)
        # Get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.isaac_gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to franka hand
        self.jacobian_eef = self.jacobian[
            :, self.franka_ee_index - 1, :, :7
        ]  # -1 due to finxed base link.
        # Prepare mass matrix tensor
        # For franka, tensor shape is (num_envs, 7 + 2, 7 + 2), 2 for grippers.
        _massmatrix = self.isaac_gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)

    def april_coord_to_sim_coord(self, april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        mat = self.april_to_sim_mat @ april_coord_mat
        return mat

    def sim_coord_to_april_coord(self, sim_coord_mat):
        mat = self.sim_to_april_mat @ sim_coord_mat
        return mat

    @property
    def april_to_sim_mat(self):
        return self.franka_from_origin_mat @ self.base_tag_from_robot_mat

    @property
    def sim_to_april_mat(self):
        return torch.tensor(
            inv(self.base_tag_from_robot_mat) @ inv(self.franka_from_origin_mat),
            device=self.device,
        )

    @property
    def sim_to_robot_mat(self):
        return torch.tensor(self.franka_from_origin_mat, device=self.device)

    @property
    def april_to_robot_mat(self):
        return torch.tensor(self.base_tag_from_robot_mat, device=self.device)

    @property
    def robot_to_ee_mat(self):
        return torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=self.device)

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs, self.pose_dim + 1))

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
                        "joint_positions": spaces.Box(low=low, high=high, shape=(7,)),
                        "joint_velocities": spaces.Box(low=low, high=high, shape=(7,)),
                        "joint_torques": spaces.Box(low=low, high=high, shape=(7,)),
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
                    low=low,
                    high=high,
                    shape=(len(self.furniture.parts) * self.pose_dim,),
                ),
            }
        )

    # @set_frequency(config['robot']['hz'])
    @torch.no_grad()
    def step(self, action):
        """Robot take action.
        Args:
            action:
                (num_envs, 8): dx, dy, dz, x, y, z, w, grip    if self.use_quat==True
                (num_envs, 7): dx, dy, dz, dax, day, daz, grip if self.quat==False
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(device=self.device)
            action.unsqueeze_(0)

        sim_steps = int(
            1.0
            / config["robot"]["hz"]
            / sim_config["sim_params"].dt
            / sim_config["sim_params"].substeps
            + 0.1
        )
        if not self.ctrl_started:
            self.init_ctrl()
        # Set the goal
        ee_pos, ee_quat = self.get_ee_pose()

        s = time.time()
        for env_idx in range(self.num_envs):
            self.osc_ctrls[env_idx].set_goal(
                action[env_idx][:3] + ee_pos[env_idx],
                C.quat_multiply(ee_quat[env_idx], action[env_idx][3:7]).to(self.device),
            )
        e = time.time()
        self.osc_times.append(e - s)

        s = time.time()
        for _ in range(sim_steps):
            self.refresh()

            pos_action = torch.zeros_like(self.dof_pos)
            torque_action = torch.zeros_like(self.dof_pos)
            grip_action = torch.zeros((self.num_envs, 1))
            for env_idx in range(self.num_envs):
                grasp = action[env_idx, -1]
                if (
                    torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                    and torch.abs(grasp) > self.grasp_margin
                ):
                    if grasp < 0:
                        grip_sep = config["robot"]["gripper_max_width"]
                    else:
                        grip_sep = 0.0
                    self.last_grasp[env_idx] = grasp
                else:
                    if self.last_grasp[env_idx] < 0:
                        grip_sep = config["robot"]["gripper_max_width"]
                    else:
                        grip_sep = 0.0

                grip_action[env_idx, -1] = grip_sep

                state_dict = {}
                ee_pos, ee_quat = self.get_ee_pose()
                state_dict["ee_pose"] = C.pose2mat(
                    ee_pos[env_idx], ee_quat[env_idx], self.device
                ).t()  # OSC expect column major
                state_dict["joint_positions"] = self.dof_pos[env_idx][:7]
                state_dict["joint_velocities"] = self.dof_vel[env_idx][:7]
                state_dict["mass_matrix"] = self.mm[env_idx][
                    :7, :7
                ].t()  # OSC expect column major
                state_dict["jacobian"] = self.jacobian_eef[
                    env_idx
                ].t()  # OSC expect column major
                torque_action[:, :7] = self.osc_ctrls[env_idx](state_dict)[
                    "joint_torques"
                ]
            # Gripper action
            pos_action[:, 7:9] = grip_action
            self.isaac_gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(pos_action)
            )
            self.isaac_gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torque_action)
            )
            # update viewer
            if not self.headless:
                self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
                self.isaac_gym.sync_frame_time(self.sim)

        self.isaac_gym.end_access_image_tensors(self.sim)
        e = time.time()

        self.ctrl_times.append(e - s)

        obs = self._get_observation()
        self.env_steps += 1

        return (
            obs,
            self._reward(),
            self._done(),
            {"obs_success": True, "action_success": True},
        )

    def _reward(self):
        # Reward is the number of newly assembled parts.
        rewards = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )

        parts_poses, founds = self._get_parts_poses()
        for env_idx in range(self.num_envs):
            env_parts_poses = parts_poses[env_idx].cpu().numpy()
            env_founds = founds[env_idx].cpu().numpy()
            rewards[env_idx] = self.furnitures[env_idx].compute_assemble(
                env_parts_poses, env_founds
            )

        if self.np_step_out:
            return rewards.cpu().numpy()

        return rewards

    def _get_parts_poses(self):
        parts_poses = torch.zeros(
            (self.num_envs, len(self.furniture.parts) * self.pose_dim),
            dtype=torch.float32,
            device=self.device,
        )
        founds = torch.ones(
            (self.num_envs, len(self.furniture.parts)),
            dtype=torch.float32,
            device=self.device,
        )
        for env_idx in range(self.num_envs):
            for part_idx in range(len(self.furniture.parts)):
                part = self.furniture.parts[part_idx]
                rb_idx = self.part_idxs[part.name][env_idx]
                part_pose = self.rb_states[rb_idx, :7]
                # To apriltag coordinate.
                part_pose = torch.concat(
                    [
                        *C.mat2pose(
                            self.sim_coord_to_april_coord(
                                C.pose2mat(
                                    part_pose[:3], part_pose[3:7], device=self.device
                                )
                            )
                        )
                    ]
                )
                parts_poses[
                    env_idx, part_idx * self.pose_dim : (part_idx + 1) * self.pose_dim
                ] = part_pose
        return parts_poses, founds

    def _save_camera_input(self):
        root = "sim_camera"
        # Make directory if not exists.
        if not os.path.exists(root):
            os.makedirs(root)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.isaac_gym.write_camera_image_to_file(
            self.sim,
            self.envs[0],
            self.camera_handles[0][0],
            gymapi.IMAGE_COLOR,
            f"{root}/{timestamp}_wrist_sim.png",
        )
        self.isaac_gym.write_camera_image_to_file(
            self.sim,
            self.envs[0],
            self.camera_handles[0][1],
            gymapi.IMAGE_COLOR,
            f"{root}/{timestamp}_front_sim.png",
        )
        self.isaac_gym.write_camera_image_to_file(
            self.sim,
            self.envs[0],
            self.camera_handles[0][2],
            gymapi.IMAGE_COLOR,
            f"{root}/{timestamp}_rear_sim.png",
        )

    def _read_robot_state(self):
        joint_positions = self.dof_pos[:, :7]
        joint_velocities = self.dof_vel[:, :7]
        joint_torques = self.forces
        ee_pos, ee_quat = self.get_ee_pose()
        for q in ee_quat:
            if q[3] < 0:
                q *= -1
        ee_pos_vel = self.rb_states[self.ee_idxs, 7:10]
        ee_ori_vel = self.rb_states[self.ee_idxs, 10:]
        gripper_width = self.gripper_width()

        return PandaState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            ee_pos_vel=ee_pos_vel,
            ee_ori_vel=ee_ori_vel,
            gripper_width=gripper_width,
        )

    def refresh(self):
        self.isaac_gym.simulate(self.sim)
        self.isaac_gym.fetch_results(self.sim, True)
        self.isaac_gym.step_graphics(self.sim)
        # refresh tensors
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)
        self.isaac_gym.refresh_mass_matrix_tensors(self.sim)
        self.isaac_gym.render_all_camera_sensors(self.sim)
        self.isaac_gym.start_access_image_tensors(self.sim)

    def init_ctrl(self):
        # Positinoal and velocity gains for robot control.
        kp = torch.tensor([90, 90, 90, 70.0, 60.0, 70.0], device=self.device)
        kv = torch.ones((6,), device=self.device) * torch.sqrt(kp) * 2.0

        ee_pos, ee_quat = self.get_ee_pose()
        for env_idx in range(self.num_envs):
            self.osc_ctrls.append(
                OSCController(
                    ee_pos_current=ee_pos[env_idx],
                    ee_quat_current=ee_quat[env_idx],
                    init_joints=torch.tensor(
                        config["robot"]["reset_joints"], device=self.device
                    ),
                    kp=kp,
                    kv=kv,
                    mass_matrix_offset_val=[0.0, 0.0, 0.0],
                    position_limits=torch.tensor(
                        config["robot"]["position_limits"], device=self.device
                    ),
                    joint_kp=10,
                )
            )
        self.ctrl_started = True

    def get_ee_pose(self):
        """Get end-effector pose in world coordinate."""
        hand_pos = self.rb_states[self.ee_idxs, :3]
        hand_quat = self.rb_states[self.ee_idxs, 3:7]
        base_pos = self.rb_states[self.base_idxs, :3]
        base_quat = self.rb_states[self.base_idxs, 3:7]  # Align with world coordinate.

        return hand_pos - base_pos, hand_quat

    def gripper_width(self):
        return self.dof_pos[:, 7:8] + self.dof_pos[:, 8:9]

    def _done(self) -> bool:
        dones = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        if self.manual_done:
            return dones
        for env_idx in range(self.num_envs):
            timeout = self.env_steps[env_idx] > self.furniture.max_env_steps
            if self.furnitures[env_idx].all_assembled() or timeout:
                dones[env_idx] = 1
                if timeout:
                    print(f"timeout {env_idx}")
        if self.np_step_out:
            dones = dones.cpu().numpy().astype(bool)
        return dones

    def _get_observation(self):
        """If successful, return (obs, True). Otherwise return (None, False)."""
        robot_state = self._read_robot_state()

        # This is world coordinate.
        parts_poses, _ = self._get_parts_poses()
        # RGBA -> RGB
        color_img1 = torch.stack(self.camera_color1)[..., :-1]
        color_img2 = torch.stack(self.camera_color2)[..., :-1]
        color_img3 = torch.stack(self.camera_color3)[..., :-1]
        if self.channel_first:
            color_img1 = color_img1.permute(0, 3, 1, 2)
            color_img2 = color_img2.permute(0, 3, 1, 2)
            color_img3 = color_img3.permute(0, 3, 1, 2)

        depth_img1 = torch.stack(self.camera_depth1)
        depth_img2 = torch.stack(self.camera_depth2)
        depth_img3 = torch.stack(self.camera_depth3)

        if self.record:
            image = color_img2
            img = image[0].permute(1, 2, 0).cpu().numpy()[..., ::-1]

            self.out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if self.np_step_out:
            color_img1 = color_img1.cpu().numpy()
            depth_img1 = depth_img1.cpu().numpy()
            color_img2 = color_img2.cpu().numpy()
            depth_img2 = depth_img2.cpu().numpy()
            color_img3 = color_img3.cpu().numpy()
            depth_img3 = depth_img3.cpu().numpy()

            robot_state = robot_state.__dict__
            for k, v in robot_state.items():
                robot_state[k] = v.cpu().numpy()
            robot_state = PandaState(**robot_state)

        return dict(
            robot_state=robot_state.__dict__,
            color_image1=color_img1,
            depth_image1=depth_img1,
            color_image2=color_img2,
            depth_image2=depth_img2,
            color_image3=color_img3,
            depth_image3=depth_img3,
            parts_poses=parts_poses,
        )

    def reset(self):
        for i in range(self.num_envs):
            self.reset_env(i)

        self.furniture.reset()

        self.refresh()
        self.assemble_idx = 0

        if self.save_camera_input:
            self._save_camera_input()

        return self._get_observation()

    def reset_env(self, env_idx):
        self.furnitures[env_idx].reset()
        # if self.randomness != Randomness.LOW_RANDOM:
        #     self.furnitures[env_idx].randomize_init_pose(self.from_skill)
        # elif self.randomness == Randomness.HIGH_RANDOM:
        #     self.furniture.randomize_high(self.high_random_idx)
        # if self.randomness == Randomness.MEDIUM_RANDOM:
        #     self.furniture.randomize_init_pose(self.from_skill)
        # elif self.randomness == Randomness.SKILL_RANDOM:
        #     self.furniture.randomize_skill_init_pose(self.from_skill)
        if self.randomness == Randomness.MEDIUM_RANDOM:
            self.furnitures[env_idx].randomize_init_pose(self.from_skill)
        elif self.randomness == Randomness.HIGH_RANDOM:
            self.furnitures[env_idx].randomize_high(self.high_random_idx)

        self._reset_franka(env_idx)
        self._reset_parts(env_idx)
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def _reset_franka(self, env_idx):
        # self.robot_model.inverse_kinematics(torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0, 1]))

        # Low randomness only.
        if self.from_skill >= 1:
            dof_pos = torch.from_numpy(self.default_dof_pos)
            ee_pos = torch.from_numpy(
                self.furniture.furniture_conf["ee_pos"][self.from_skill]
            )
            ee_quat = torch.from_numpy(
                self.furniture.furniture_conf["ee_quat"][self.from_skill]
            )
            dof_pos = self.robot_model.inverse_kinematics(ee_pos, ee_quat)
        else:
            dof_pos = self.default_dof_pos

        self.dof_pos[:, 0 : self.franka_num_dofs] = torch.tensor(
            dof_pos, device=self.device, dtype=torch.float32
        )
        self.dof_vel[:, 0 : self.franka_num_dofs] = torch.tensor(
            [0] * len(self.default_dof_pos), device=self.device, dtype=torch.float32
        )
        idxs = torch.tensor(self.franka_handles, device=self.device, dtype=torch.int32)[
            env_idx : env_idx + 1
        ]

        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(idxs),
            len(idxs),
        )

    def _reset_parts(self, env_idx, random_noise_fixed=False):
        """Reset the parts to the initial pose.
        Args:
            env_idx (int): The index of the environment.
            random_noise_fixed (bool): If True, add random noise to the fixed initial pose.
        """
        for part in self.furnitures[env_idx].parts:
            pos, ori = self._get_reset_pose(part)

            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = gymapi.Transform()
            part_pose.p = gymapi.Vec3(
                part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
            )
            if self.randomness == Randomness.LOW_RANDOM and random_noise_fixed:
                part_pose.p = part_pose.p + gymapi.Vec3(
                    random.uniform(-0.015, 0.015), random.uniform(-0.015, 0.015), 0
                )  # 1.5 cm

            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
            if self.randomness == Randomness.LOW_RANDOM and random_noise_fixed:
                part_pose.r = (
                    gymapi.Quat.from_axis_angle(
                        gymapi.Vec3(0, 0, 1), np.radians(np.random.uniform(-15, 15))
                    )
                    * part_pose.r
                )  # 15 degrees. noise
            idxs = self.parts_handles[part.name]
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.int32)

            self.root_pos[env_idx, idxs] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.root_quat[env_idx, idxs] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )

        idxs = torch.tensor(
            list(self.parts_handles.values()), dtype=torch.int32, device=self.device
        )
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(idxs),
            len(idxs),
        )

    def _import_base_tag_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_asset_file = f"furniture/urdf/base_tag.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, base_asset_file, asset_options
        )

    def _import_obstacle_front_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = f"furniture/urdf/obstacle_front.urdf"
        obstacle_front_asset = self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

        return obstacle_front_asset

    def _import_obstacle_side_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_side.urdf"
        obstacle_side_asset = self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

        return obstacle_side_asset

    def _import_background_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        background_asset_file = "furniture/urdf/background.urdf"

        background_asset = self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, background_asset_file, asset_options
        )
        return background_asset

    def _import_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset_file = "furniture/urdf/table.urdf"

        table_asset = self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, table_asset_file, asset_options
        )
        return table_asset

    def _import_franka_asset(self):
        # Franka asset.
        self.franka_asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda.urdf"
        )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, self.franka_asset_file, asset_options
        )

        return franka_asset

    def get_assembly_action(self) -> torch.Tensor:
        """
        Returns: Tuple (action for the assembly task, skill complete mask)
        """
        assert self.num_envs == 1  # Only support one environment for now.
        if self.assemble_idx > len(self.furniture.should_be_assembled):
            return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=self.device)

        ee_pos, ee_quat = self.get_ee_pose()
        gripper_width = self.gripper_width()
        ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()

        if self.move_neutral:
            if ee_pos[2] <= 0.15 - 0.01:
                gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)
                goal_pos = torch.tensor(
                    [ee_pos[0], ee_pos[1], 0.15], device=self.device
                )
                delta_pos = goal_pos - ee_pos
                delta_quat = torch.tensor([0, 0, 0, 1], device=self.device)
                action = torch.concat([delta_pos, delta_quat, gripper])
                return action.unsqueeze(0), 0
            else:
                self.move_neutral = False
        part_idx1, part_idx2 = self.furniture.should_be_assembled[self.assemble_idx]

        part1 = self.furniture.parts[part_idx1]
        part1_name = self.furniture.parts[part_idx1].name
        part1_pose = C.to_homogeneous(
            self.rb_states[self.part_idxs[part1_name]][0][:3],
            C.quat2mat(self.rb_states[self.part_idxs[part1_name]][0][3:7]),
        )
        part2 = self.furniture.parts[part_idx2]
        part2_name = self.furniture.parts[part_idx2].name
        part2_pose = C.to_homogeneous(
            self.rb_states[self.part_idxs[part2_name]][0][:3],
            C.quat2mat(self.rb_states[self.part_idxs[part2_name]][0][3:7]),
        )
        rel_pose = torch.linalg.inv(part1_pose) @ part2_pose
        assembled_rel_poses = self.furniture.assembled_rel_poses[(part_idx1, part_idx2)]
        if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
            self.assemble_idx += 1
            self.move_neutral = True
            return (
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 1, -1], dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                1,
            )  # Skill complete is always 1 when assembled.
        if not part1.pre_assemble_done:
            goal_pos, goal_ori, gripper, skill_complete = part1.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
            )
        elif not part2.pre_assemble_done:
            goal_pos, goal_ori, gripper, skill_complete = part2.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
            )
        else:
            goal_pos, goal_ori, gripper, skill_complete = self.furniture.parts[
                part_idx2
            ].fsm_step(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
                self.furniture.parts[part_idx1].name,
            )

        delta_pos = goal_pos - ee_pos

        # Scale translational action.
        delta_pos_sign = delta_pos.sign()
        delta_pos = torch.abs(delta_pos) * 2
        for i in range(3):
            if delta_pos[i] > 0.03:
                delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
        delta_pos = delta_pos * delta_pos_sign

        # Clamp too large action.
        max_delta_pos = 0.11 + 0.01 * torch.rand(3, device=self.device)
        max_delta_pos[2] -= 0.04
        delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)

        delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
        # q0 = torch.tensor([0., 0., 0., 1.], device=self.device)
        # t = torch.tensor((1.5, ), device=self.device)
        # delta_quat = roma.utils.unitquat_slerp(q0, delta_quat, t)[0]

        # Add random noise to the action.
        if (
            self.furniture.parts[part_idx2].state_no_noise()
            and random.uniform(0, 1) < 0.50
        ):
            delta_pos = torch.normal(delta_pos, 0.005)
            delta_quat = C.quat_multiply(
                delta_quat,
                torch.tensor(
                    T.axisangle2quat(
                        [
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                        ]
                    ),
                    device=self.device,
                ),
            ).to(self.device)
        action = torch.concat([delta_pos, delta_quat, gripper])
        return action.unsqueeze(0), skill_complete

    def assembly_success(self):
        return self._done().squeeze()

    def __del__(self):
        if not self.headless:
            self.isaac_gym.destroy_viewer(self.viewer)
        self.isaac_gym.destroy_sim(self.sim)
        if self.record:
            self.out.release()
