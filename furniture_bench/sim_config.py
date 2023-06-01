"""Define additional parameters based on real-world config for simulator."""

from isaacgym import gymapi

from furniture_bench.config import config

sim_config = config.copy()

# Simulator options.
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = True
sim_params.physx.solver_type = 1
sim_params.physx.bounce_threshold_velocity = 0.02
sim_params.physx.num_position_iterations = 20
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.0
sim_params.physx.contact_offset = 0.002
sim_params.physx.friction_offset_threshold = 0.01
sim_params.physx.friction_correlation_distance = 0.0005
sim_params.physx.use_gpu = True

sim_config["sim_params"] = sim_params
sim_config["parts"] = {"friction": 0.15}
sim_config["table"] = {"friction": 0.15}
sim_config["asset"] = {}

# Parameters for the robot.
sim_config["robot"].update(
    {
        "kp": [90, 90, 90, 70.0, 60.0, 80.0],  # Default positional gains.
        "kv": None,  # Default velocity gains.
        "arm_frictions": [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],  # Default arm friction.
        "gripper_frictions": [15.0, 15.0],  # Default gripper friction.
        "gripper_torque": 13,  # Default torque for gripper.
    }
)

# Parameters for the light.
sim_config["lights"] = [
    {
        "color": [0.8, 0.8, 0.8],
        "ambient": [0.35, 0.35, 0.35],
        "direction": [0.1, -0.03, 0.2],
    }
]

"""
Set density for each furniture part.
  - The volume is estimated using Belnder.
  - The mass is estimated using 3D printer slicer.
"""


def default_asset_options():
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = False
    asset_options.thickness = 0.0
    asset_options.density = 600.0
    # asset_options.armature = 0.01
    asset_options.linear_damping = 0.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.angular_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.disable_gravity = False
    asset_options.enable_gyroscopic_forces = True

    return asset_options


# Volume: 302802 mm^3
# Mass: 151g
square_table_top_asset_options = default_asset_options()
square_table_top_asset_options.density = 498.68
sim_config["asset"]["square_table_top"] = square_table_top_asset_options

# Volume: 62435.mm^3
# Mass: 23.1g
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["square_table_leg1"] = leg_asset_options
sim_config["asset"]["square_table_leg2"] = leg_asset_options
sim_config["asset"]["square_table_leg3"] = leg_asset_options
sim_config["asset"]["square_table_leg4"] = leg_asset_options

# Cabinet.
# Volume: 224623 mm^3
# Mass: 130.98g
cabinet_body_asset_options = default_asset_options()
cabinet_body_asset_options.density = 583.11
sim_config["asset"]["cabinet_body"] = cabinet_body_asset_options

# Volume: 73208 mm^3
# Mass: 30.2g
cabinet_door_left_asset_options = default_asset_options()
cabinet_door_left_asset_options.density = 412.52
sim_config["asset"]["cabinet_door_left"] = cabinet_door_left_asset_options
sim_config["asset"]["cabinet_door_right"] = cabinet_door_left_asset_options

# Volume: 192689 mm^3
# Mass: 60.29g
cabinet_top_asset_options = default_asset_options()
cabinet_top_asset_options.density = 312.89
sim_config["asset"]["cabinet_top"] = cabinet_top_asset_options

# Desk.
# Volume: 343624 mm^3
# Mass: 169.4g
desk_top_asset_options = default_asset_options()
desk_top_asset_options.density = 492.98
sim_config["asset"]["desk_top"] = desk_top_asset_options

# Volume: 181892 mm^3
# Mass: 56.2g
desk_leg1_asset_options = default_asset_options()
desk_leg1_asset_options.density = 308.92
sim_config["asset"]["desk_leg1"] = desk_leg1_asset_options
sim_config["asset"]["desk_leg2"] = desk_leg1_asset_options
sim_config["asset"]["desk_leg3"] = desk_leg1_asset_options
sim_config["asset"]["desk_leg4"] = desk_leg1_asset_options

# Round table.
# Volume: 257631 mm^3
# Mass: 121.69g
round_table_top_asset_options = default_asset_options()
round_table_top_asset_options.density = 472.34
sim_config["asset"]["round_table_top"] = round_table_top_asset_options

# Volume: 75321 mm^3
# Mass:  32.28g
round_table_leg_asset_options = default_asset_options()
round_table_leg_asset_options.density = 414.52
sim_config["asset"]["round_table_leg"] = round_table_leg_asset_options

# Volume: 81926 mm^3
# Mass: 33.96g
round_table_base_asset_options = default_asset_options()
round_table_base_asset_options.density = 533.11
sim_config["asset"]["round_table_base"] = round_table_base_asset_options

# Drawer
# Volume: 221853 mm^3
# Mass: 151.63g
drawer_box_asset_options = default_asset_options()
drawer_box_asset_options.density = 683.47
sim_config["asset"]["drawer_box"] = drawer_box_asset_options

# Volume:  92893 mm^3
# Mass: 59.37g
drawer_container_top_asset_options = default_asset_options()
drawer_container_top_asset_options.density = 639.1
sim_config["asset"]["drawer_container_top"] = drawer_container_top_asset_options
sim_config["asset"]["drawer_container_bottom"] = drawer_container_top_asset_options

# Chair
# Volume: 111594 mm^3
# MAss: 61.87g
chair_seat_asset_options = default_asset_options()
chair_seat_asset_options.density = 554.42
sim_config["asset"]["chair_seat"] = chair_seat_asset_options

# Volume: 354703 mm^3
# Mass: 123.16g
chair_back_asset_options = default_asset_options()
chair_back_asset_options.density = 347.22
sim_config["asset"]["chair_back"] = chair_back_asset_options

# Volume: 60139 mm^3
# MAss: 22.44g
chair_leg1_asset_options = default_asset_options()
chair_leg1_asset_options.density = 373.14
sim_config["asset"]["chair_leg1"] = chair_leg1_asset_options
sim_config["asset"]["chair_leg2"] = chair_leg1_asset_options

# Volume: 20083 mm^3
# Mass: 10.15g
chair_nut1_asset_options = default_asset_options()
chair_nut1_asset_options.density = 505.40
sim_config["asset"]["chair_nut1"] = chair_nut1_asset_options
sim_config["asset"]["chair_nut2"] = chair_nut1_asset_options

# Lamp
# Volume:  78694 mm^3
# Mass: 59.99g
lamp_hood_asset_options = default_asset_options()
lamp_hood_asset_options.density = 762.31
sim_config["asset"]["lamp_hood"] = lamp_hood_asset_options

# Volume:  174649 mm^3
# Mass: 59.65g
lamp_base_asset_options = default_asset_options()
lamp_base_asset_options.density = 341.54
sim_config["asset"]["lamp_base"] = lamp_base_asset_options

# Volume: 70576 mm^3
# Mass: 38.47g
lamp_bulb_asset_options = default_asset_options()
lamp_bulb_asset_options.density = 545.09
sim_config["asset"]["lamp_bulb"] = lamp_bulb_asset_options

# Stool
# Volume: 103515 mm^3
# Mass: 57.34g
stool_seat_asset_options = default_asset_options()
stool_seat_asset_options.density = 553.93
sim_config["asset"]["stool_seat"] = stool_seat_asset_options

# Volume: 81131 mm^3
# Mass: 27.07g
stool_leg1_asset_options = default_asset_options()
stool_leg1_asset_options.density = 333.66
sim_config["asset"]["stool_leg1"] = stool_leg1_asset_options
sim_config["asset"]["stool_leg2"] = stool_leg1_asset_options
sim_config["asset"]["stool_leg3"] = stool_leg1_asset_options
