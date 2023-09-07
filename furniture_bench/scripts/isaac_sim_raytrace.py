# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import argparse
import time
import os
from datetime import datetime

from omni.isaac.kit import SimulationApp
from furniture_bench.utils.pose import get_mat, rot_mat

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless",
                    action="store_true",
                    default=False,
                    help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": args_cli.headless})
"""Rest everything follows."""

import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
import omni.replicator.core as rep
from omni.isaac.orbit.utils import convert_dict_to_backend

FILE = '/home/minho/furniture-bench/scripted_sim_demo/one_leg/2023-09-05-16:07:41/2023-09-05-16:07:41.pkl'

import numpy as np

import furniture_bench.utils.transform as T
from furniture_bench.config import config

april_to_sim_mat = np.array([[6.1232343e-17, 1.0000000e+00, 1.2246469e-16, 1.4999807e-03],
                             [1.0000000e+00, -6.1232343e-17, -7.4987988e-33, 0.0000000e+00],
                             [0.0000000e+00, 1.2246469e-16, -1.0000000e+00, 4.1500002e-01],
                             [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])


def gen_prim(usd_path: str, prim_name: str, init_pos, init_ori):
    furniture_assets_root_path = f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/'
    usd_path = furniture_assets_root_path + usd_path
    pose = april_to_sim_mat @ (T.to_homogeneous(init_pos, init_ori))
    pos, ori = T.mat2pose(pose)
    ori = T.convert_quat(ori, to='wxyz')

    prim_utils.create_prim(prim_name, usd_path=usd_path, translation=pos, orientation=ori)

    view = RigidPrimView(prim_name, reset_xform_properties=False)
    return view


def main():
    s = time.time()
    with open(FILE, 'rb') as f:
        data = pickle.load(f)
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera

    # Setup camera sensor
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device='cpu')

    # Spawn camera
    camera.spawn("/World/CameraSensor")

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 5.0),
        attributes={
            "radius": 4.5,
            "intensity": 1200.0,
            "color": (0.75, 0.75, 0.75)
        },
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 5.0),
        attributes={
            "radius": 4.5,
            "intensity": 1200.0,
            "color": (1.0, 1.0, 1.0)
        },
    )
    # Doom linght
    # prim_utils.create_prim(
    #     "/World/Light/DoomLight",
    #     "DoomLight",
    #     translation=(-4.5, 3.5, 10.0),
    #     attributes={
    #         "radius": 4.5,
    #         "intensity": 300.0,
    #         "color": (1.0, 1.0, 1.0)
    #     },
    # )

    # Table
    # table_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

    table_pos = (0., 0., 0.4)
    prim_utils.create_prim(
        "/World/Table",
        usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/table.usd',
        translation=table_pos)

    views = []

    # Background
    prim_utils.create_prim(
        '/World/Background',
        translation=(-0.8, 0, 0.75),
        usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/background.usd')

    # Base tag
    prim_utils.create_prim(
        '/World/BaseTag',
        translation=(0, 0, 0.415),
        usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/base_tag.usd')

    # Obstacle front.
    prim_utils.create_prim(
        '/World/ObstacleFront',
        translation=(0.3815, 0., 0.43),
        orientation=(0.7071067690849304, 0, 0, 0.7071067690849304),
        usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/obstacle_front.usd')

    # Obstacle side.
    prim_utils.create_prim(
        '/World/ObstacleSide1',
        translation=(0.306, -0.175, 0.43),
        orientation=(0.7071067690849304, 0, 0, 0.7071067690849304),
        usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/obstacle_side.usd')
    prim_utils.create_prim(
        '/World/ObstacleSide2',
        translation=(0.306, 0.175, 0.43),
        orientation=(0.7071067690849304, 0, 0, 0.7071067690849304),
        usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/obstacle_side.usd')

    # bg_rot = rot_mat([0, 0, np.pi / 2])
    # bg_view = gen_prim('background.usd', , torch.tensor([0.8, 0, 0.75]), bg_rot)

    square_table_top_pos = config['furniture']['square_table']['square_table_top']['reset_pos'][0]
    square_table_top_ori = config['furniture']['square_table']['square_table_top']['reset_ori'][
        0][:3, :3]
    view = gen_prim('square_table/square_table_top.usd', '/World/SquareTableTop',
                    square_table_top_pos, square_table_top_ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg1']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg1']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg1.usd', '/World/SquareTableLeg1', pos, ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg2']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg2']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg2.usd', '/World/SquareTableLeg2', pos, ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg3']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg3']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg3.usd', '/World/SquareTableLeg3', pos, ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg4']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg4']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg4.usd', '/World/SquareTableLeg4', pos, ori)
    views.append(view)

    # Robots
    # -- Resolve robot config from command-line arguments
    if args_cli.robot == "franka_panda":
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    elif args_cli.robot == "ur10":
        robot_cfg = UR10_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # -- Spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    franka_from_origin_mat = np.array([[1., 0., 0., -0.3], [0., 1., 0., 0.], [0., 0., 1., 0.43],
                                       [0., 0., 0., 1.]])
    pos, ori = T.mat2pose(franka_from_origin_mat)
    robot.spawn("/World/Robot_2", translation=pos)

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/Robot.*")
    # Reset states
    robot.reset_buffers()

    # Initialize camera
    camera.initialize()

    cam_pos = (1.3, -0.00, 0.80)
    cam_target = (-1, -0.00, 0.4)
    camera.set_world_pose_from_view(cam_pos, cam_target)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics

    sim_steps = 1.0 / 10 / sim_dt  # #steps / 10Hz / dt

    # Initialize robot state
    robot.set_dof_state(
        torch.concat([
            torch.tensor(data['observations'][0]['robot_state']['joint_positions']),
            torch.from_numpy(np.array([data['observations'][0]['robot_state']['gripper_width']])) /
            2,
            torch.from_numpy(np.array([data['observations'][0]['robot_state']['gripper_width']])) /
            2
        ]),
        torch.concat([
            torch.tensor(data['observations'][0]['robot_state']['joint_velocities']),
            torch.zeros((1, )),
            torch.zeros((1, ))
        ]))

    prev_goal_pos = torch.tensor(data['observations'][0]['robot_state']['joint_positions'])

    parts_prev_goal_pos = []
    parts_prev_goal_ori = []
    part_idx_offset = 1

    prev_gripper_width = data['observations'][0]['robot_state']['gripper_width']

    for i in range(5):
        pos = data['observations'][part_idx_offset]['parts_poses'][7 * i:7 * i + 3]
        ori = data['observations'][part_idx_offset]['parts_poses'][7 * i + 3:7 * i + 7]
        parts_prev_goal_pos.append(pos)
        parts_prev_goal_ori.append(ori)

    for _ in range(100):
        sim.step()
        time.sleep(0.01)

    for obs_idx, obs in enumerate(data['observations']):
        # while True:
        goal_pos = torch.tensor(obs['robot_state']['joint_positions'])
        dx = (goal_pos - prev_goal_pos) / sim_steps

        part_idx = obs_idx + part_idx_offset if obs_idx + part_idx_offset < len(
            data['observations']) else len(data['observations']) - 1

        for i in range(int(sim_steps)):
            interp_goal = prev_goal_pos + (i + 1) * dx

            # If simulation is paused, then skip.
            if not sim.is_playing():
                sim.step(render=not args_cli.headless)
                continue

            # Update camera data
            camera.update(dt=0.0)

            rep_writer.write(convert_dict_to_backend(camera.data.output, backend="numpy"))

            griper_dx = (obs['robot_state']['gripper_width'] - prev_gripper_width) / sim_steps
            gripper_interp_goal = prev_gripper_width + (i + 1) * griper_dx
            robot.set_dof_state(
                torch.concat([
                    interp_goal,
                    torch.from_numpy(np.array([gripper_interp_goal])).float() / 2,
                    torch.from_numpy(np.array([gripper_interp_goal])).float() / 2
                ]),
                torch.concat([
                    torch.tensor(obs['robot_state']['joint_velocities']),
                    torch.zeros((1, )),
                    torch.zeros((1, ))
                ]))

            for j in range(5):
                # pos = obs['parts_poses'][7 * i:7 * i + 3]
                goal_pos = data['observations'][part_idx]['parts_poses'][7 * j:7 * j + 3]
                part_dx = (goal_pos - parts_prev_goal_pos[j]) / sim_steps
                pos = torch.tensor(parts_prev_goal_pos[j] + (i + 1) * part_dx)

                goal_ori = data['observations'][part_idx]['parts_poses'][7 * j + 3:7 * j + 7]
                interp_fraction = i / sim_steps
                ori = T.quat_slerp(parts_prev_goal_ori[j], goal_ori, fraction=interp_fraction)

                rot = T.quat2mat(ori)
                pose = april_to_sim_mat @ (T.to_homogeneous(pos, rot))
                pos, ori = T.mat2pose(pose)
                ori = T.convert_quat(ori, to='wxyz')
                pos = torch.from_numpy(pos).unsqueeze(0)
                ori = torch.from_numpy(ori).unsqueeze(0)
                views[j].set_world_poses(positions=pos, orientations=ori)

            # perform step
            sim.step()
            # update sim-time
            sim_time += sim_dt
            ep_step_count += 1
            # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
            if sim.is_playing():
                # update buffers
                robot.update_buffers(sim_dt)


        prev_goal_pos = torch.tensor(obs['robot_state']['joint_positions'])
        prev_gripper_width = obs['robot_state']['gripper_width']
        parts_prev_goal_pos = []
        parts_prev_goal_ori = []
        for i in range(5):
            pos = data['observations'][part_idx]['parts_poses'][7 * i:7 * i + 3]
            ori = data['observations'][part_idx]['parts_poses'][7 * i + 3:7 * i + 7]
            parts_prev_goal_pos.append(pos)
            parts_prev_goal_ori.append(ori)

    e=time.time()
    print(f"Time taken: {e-s}")

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
