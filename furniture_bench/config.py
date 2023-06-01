"""Configuration for environment, robot, furniture, and camera."""
import os
from typing import Any, Dict

import numpy as np

from furniture_bench.utils.pose import get_mat, rot_mat

ROBOT_HEIGHT = 0.015  # Approximate height of bench clamp: 1.5 cm.

# Fill in information below or define environment variables.
SERVER_IP = os.getenv("SERVER_IP", "")
CAM_WRIST_SERIAL = os.getenv("CAM_WRIST_SERIAL", "")
CAM_FRONT_SERIAL = os.getenv("CAM_FRONT_SERIAL", "")
CAM_REAR_SERIAL = os.getenv("CAM_REAR_SERIAL", "")


config: Dict[str, Any] = {
    "robot": {
        "server_ip": SERVER_IP,
        "hz": 10,
        "reset_joints": [
            -0.02630888,
            0.3758795,
            0.12485036,
            -2.1383357,
            -0.09431414,
            2.49649072,
            0.01921718,
        ],
        "base_tag_xyz": (0.23 + 0.0715, 0, -ROBOT_HEIGHT),
        "tag_base_from_robot_base": get_mat(
            (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
        ),  # Relative pose of the base tag from the robot base. This can be used to convert the pose in the base tag's coordinate to the robot's coordinate.
        "max_gripper_width": {
            "square_table": 0.065,
            "one_leg": 0.065,
            "desk": 0.065,
            "stool": 0.065,
            "chair": 0.065,
            "drawer": 0.065,
            "round_table": 0.065,
            "cabinet": 0.08,
            "lamp": 0.07,
        },
        "position_limits": [
            [0.3, 0.8],
            [-0.55, 0.55],
            [0.005, 0.4],
        ],  # End-effector position limits.
        "pos_noise_med": 0.05,  # Positional noise for medium randomness.
        "rot_noise_med": 15,  # Rotational noise for medium randomness.
        "motion_stopped_counter_threshold": 50,  # Number of steps to wait when the robot stopped moving before declaring the episode done.
    },
    "camera": {
        "num_camera": 3,
        1: {
            "serial": CAM_WRIST_SERIAL,
            "rel_pose_from_base": get_mat(
                [0.001, 0.7, -0.28], [np.pi / 2 - np.pi / 9, 0, 0]
            ),
        },
        2: {
            "serial": CAM_FRONT_SERIAL,
            "rel_pose_from_base": get_mat(
                [-0.25, -0.13, -0.31], [np.pi / 2 - np.pi / 4, 0, -np.pi - np.pi / 6]
            ),
            "roi": [220, 1020, 220, 690],  # [min_x, max_x, min_y, max_y]
        },
        3: {
            "serial": CAM_REAR_SERIAL,
            "rel_pose_from_base": get_mat(
                [0.25, -0.13, -0.31], [np.pi / 2 - np.pi / 4, 0, -np.pi + np.pi / 6]
            ),
        },
        "color_img_size": (1280, 720),
        "depth_img_size": (1280, 720),
        "resized_img_size": (224, 224),
        "frame_rate": 30,
    },
    "furniture": {
        "detection_hz": 30,
        "base_tags": [0, 1, 2, 3],
        "base_tag_size": 0.048,
        "action_dim": 3
        + 4
        + 1,  # (xyz position delta, delta quaternion, gripper action)
        "env_img_size": (224, 224),
        "position_limits": [
            [-0.21, 0.21],
            [0.07, 0.37],
        ],  # Furniture parts position limits.
        "rel_pose_from_coordinate": {
            0: get_mat([-0.03, -0.03, 0], [0, 0, 0]),
            1: get_mat([0.03, -0.03, 0], [0, 0, 0]),
            2: get_mat([-0.03, 0.03, 0], [0, 0, 0]),
            3: get_mat([0.03, 0.03, 0], [0, 0, 0]),
        },
        "reset_temps": [(0, 0.03, 0)],
        "square_table": {
            "tag_size": 0.0195,
            "total_reward": 4.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.45, -0.015, 0.01], dtype=np.float32),
                np.array([0.52, 0.07, 0.035], dtype=np.float32),
                np.array([0.42, 0.18, 0.015], dtype=np.float32),
                np.array([0.65, 0.12, 0.08], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.70579994, -0.7077995, 0.00673244, 0.02865304], dtype=np.float32
                ),
                np.array(
                    [-0.71581334, -0.6981131, -0.00658566, 0.01435379], dtype=np.float32
                ),
                np.array(
                    [0.89031214, 0.00794221, -0.4523126, 0.05190935], dtype=np.float32
                ),
                np.array(
                    [-0.9996031, 0.00642258, 0.01763591, 0.0210082], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "square_table_top": {
                "name": "square_table_top",
                "asset_file": "furniture/urdf/square_table/square_table_top.urdf",
                "ids": [4, 5, 6, 7],
                "reset_pos": [
                    np.array([0.0, 0.24, -0.015625]),
                    np.array([0.0, 0.24, -0.015625]),
                    np.array([0.08, 0.27, -0.015], dtype=np.float32),
                    np.array([0.08, 0.27, -0.015625], dtype=np.float32),
                    np.array([0.08, 0.26, -0.015625], dtype=np.float32),
                ],
                "reset_ori": [
                    rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True),
                    rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True),
                    np.array(
                        [
                            [-9.9997163e-01, -9.2363916e-07, -7.5414525e-03, 0],
                            [7.5414525e-03, -3.8682265e-07, -9.9997157e-01, 0],
                            [9.2084520e-07, -1.0000000e00, 3.2843309e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            [-9.9997163e-01, -9.2363916e-07, -7.5414525e-03, 0],
                            [7.5414525e-03, -3.8682265e-07, -9.9997157e-01, 0],
                            [9.2084520e-07, -1.0000000e00, 3.2843309e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            [-9.9997151e-01, -6.1001629e-07, -7.5414428e-03, 0],
                            [7.5414428e-03, 2.0808420e-07, -9.9997151e-01, 0],
                            [6.1141327e-07, -1.0000000e00, -1.4954367e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                ],
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([0.00165764, 0.23166147, -0.01571113])],
                    [np.array([-0.02074748, 0.20895012, -0.01565138])],
                    [np.array([-0.07365019, 0.13859308, -0.01718625])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.22164249, -0.00542635, -0.97511292, 0.0],
                                [0.97511977, -0.00533661, -0.22161436, 0.0],
                                [-0.00400126, -0.99997103, 0.00647417, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.18759322, -0.01144737, -0.98218012, 0.0],
                                [0.98217165, -0.01018527, 0.18771034, 0.0],
                                [-0.01215252, -0.99988264, 0.00933257, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.34981984, -0.00092953, -0.93681651, 0.0],
                                [0.93678069, -0.00914782, -0.34979743, 0.0],
                                [-0.00824469, -0.99995774, 0.00407088, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "square_table_leg1": {
                "name": "square_table_leg1",
                "asset_file": "furniture/urdf/square_table/square_table_leg1.urdf",
                "ids": [8, 9, 10, 11],
                "reset_pos": [np.array([-0.20, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "half_width": 0.015,
                "default_assembled_pose": get_mat(
                    np.array([0.05625, 0.046875, 0.05625]), [0, 0, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.12411077, 0.172485, -0.01282854])],
                    [np.array([0.11414307, 0.16866885, -0.01511348])],
                    [np.array([0.07065351, 0.16127343, -0.0162864])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.99050552, 0.13746983, -0.00065276, 0.0],
                                [0.13745576, 0.9904536, 0.01037287, 0.0],
                                [0.00207249, 0.01018466, -0.99994576, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.99296767, -0.11772603, 0.01248006, 0.0],
                                [0.11803517, 0.99261737, -0.02790101, 0.0],
                                [-0.00910325, 0.02917789, 0.99953276, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.00333085, -0.52690697, -0.8499164, 0.0],
                                [-0.01105705, 0.84984976, -0.52690899, 0.0],
                                [0.9999333, 0.01115263, -0.00299532, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "square_table_leg2": {
                "name": "square_table_leg2",
                "asset_file": "furniture/urdf/square_table/square_table_leg2.urdf",
                "ids": [12, 13, 14, 15],
                "reset_pos": [np.array([-0.12, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "half_width": 0.015,
                "default_assembled_pose": get_mat(
                    np.array([-0.05625, 0.046875, 0.05625]), [0, 0, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.14908047, 0.14851391, -0.02330433])],
                    [np.array([-0.04624121, 0.20801648, -0.01895902])],
                    [np.array([-0.07811823, 0.31678003, -0.01311489])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.98727685, 0.15897338, 0.0034444, 0.0],
                                [-0.15894894, 0.98726553, -0.00647893, 0.0],
                                [-0.00443052, 0.00584901, 0.99997306, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.9999948, 0.0032234, 0.00009178, -0.04623413],
                                [-0.00322341, 0.9999948, 0.00006646, 0.20785046],
                                [-0.00009156, -0.00006676, 1.0, -0.02099609],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.00119546, -0.40191379, -0.91567671, 0.0],
                                [-0.00777176, 0.91565347, -0.40189344, 0.0],
                                [0.99996912, 0.00663596, -0.0042182, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "square_table_leg3": {
                "name": "square_table_leg3",
                "asset_file": "furniture/urdf/square_table/square_table_leg3.urdf",
                "ids": [16, 17, 18, 19],
                "reset_pos": [np.array([0.12, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "half_width": 0.015,
                "default_assembled_pose": get_mat(
                    np.array([0.05625, 0.046875, -0.05625]), [0, 0, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([-0.11946367, 0.18800808, -0.01739212])],
                    [np.array([0.21110627, 0.09755837, -0.01618727])],
                    [np.array([0.1526014, 0.10060443, -0.0157905])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.01415714, -0.15679362, 0.98752993, 0.0],
                                [0.02789432, 0.98718256, 0.15713838, 0.0],
                                [-0.99951059, 0.0297711, -0.00960203, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.00594478, -0.18838574, -0.98207718, 0.0],
                                [-0.01028048, 0.98203105, -0.18843913, 0.0],
                                [0.99992949, 0.01121645, 0.00390126, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.98098826, -0.19385707, -0.00902952, 0.0],
                                [-0.19405723, 0.98034805, 0.03548874, 0.0],
                                [0.00197233, 0.03656628, -0.99932933, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "square_table_leg4": {
                "name": "square_table_leg4",
                "asset_file": "furniture/urdf/square_table/square_table_leg4.urdf",
                "ids": [20, 21, 22, 23],
                "reset_pos": [
                    np.array([0.20, 0.07, -0.015]),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.136, 0.336, -0.07763672], dtype=np.float32),
                ],
                "reset_ori": [
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    np.array(
                        [
                            [0.09753844, 0.03445375, -0.9946352, 0],
                            [0.9915855, 0.08210686, 0.10008356, 0],
                            [0.08511469, -0.9960278, -0.02615529, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                ],
                "half_width": 0.015,
                "default_assembled_pose": get_mat(
                    np.array([-0.05625, 0.046875, -0.05625]), [0, 0, 0]
                ),
                "part_attached_skill_idx": 4,
                "high_rand_reset_pos": [
                    [np.array([-0.15307285, 0.12734982, -0.01829645])],
                    [np.array([-0.14806452, 0.25308317, -0.01665781])],
                    [np.array([0.08496851, 0.28050846, -0.01651655])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.79311496, -0.6090616, -0.0035405, 0.0],
                                [-0.60903323, 0.79298568, 0.01588211, 0.0],
                                [-0.00686562, 0.01475262, -0.99986762, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.88222593, -0.47080287, -0.00470781, 0.0],
                                [-0.470696, 0.88170207, 0.03235343, 0.0],
                                [-0.0110812, 0.03075899, -0.99946541, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.01009589, -0.99462211, -0.10307747, 0.0],
                                [-0.01416063, 0.10293016, -0.99458778, 0.0],
                                [0.99984878, 0.01150092, -0.0130454, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
        "one_leg": {
            "total_reward": 1.0,
        },
        "desk": {
            "tag_size": 0.0195,
            "total_reward": 4.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.48, -0.0, 0.0097], dtype=np.float32),
                np.array([0.523, 0.0333, 0.02], dtype=np.float32),
                np.array([0.438, 0.135, 0.0118], dtype=np.float32),
                np.array([0.62, 0.15, 0.12], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.7110111, -0.7030691, -0.00348992, 0.01203612], dtype=np.float32
                ),
                np.array(
                    [-0.7323884, -0.68016654, -0.03116747, 0.00304913], dtype=np.float32
                ),
                np.array(
                    [0.89031214, 0.00794221, -0.4523126, 0.05190935], dtype=np.float32
                ),
                np.array(
                    [-0.9494539, -0.0773899, -0.30140638, 0.04125888], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "desk_top": {
                "name": "desk_top",
                "asset_file": "furniture/urdf/desk/desk_top.urdf",
                "ids": [109, 110, 111, 112],
                "reset_pos": [[0, 0.24, -0.01592]] * 2 + [[0.07, 0.275, -0.01592]] * 3,
                "reset_ori": [rot_mat(np.array([-np.pi / 2, 0, 0]), hom=True)] * 5,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([-0.002, 0.104, -0.017])],
                    [np.array([-0.05966621, 0.25662723, -0.01701638])],
                    [np.array([0.02099706, 0.16332534, -0.01756242])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.943, -0.008, 0.332, 0.0],
                                [-0.332, 0.046, -0.942, 0.0],
                                [-0.008, -0.999, -0.046, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.99573445, -0.00066294, -0.09226216, 0.0],
                                [0.09208053, 0.05598222, -0.99417663, 0.0],
                                [0.00582412, -0.99843156, -0.05568231, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.98298514, -0.0125294, -0.18325743, 0.0],
                                [0.18326062, 0.0009035, -0.983064, 0.0],
                                [0.01248278, -0.99992114, 0.00140799, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "leg1": {
                "name": "desk_leg1",
                "asset_file": "furniture/urdf/desk/desk_leg1.urdf",
                "ids": [113, 114, 115, 116],
                "reset_pos": [[-0.15, 0.07, -0.0175]] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "half_width": 0.0175,
                "default_assembled_pose": get_mat(
                    np.array([-0.080, 0.065625, -0.050001]), [0, -np.pi / 4, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([-0.118, 0.227, -0.019])],
                    [np.array([-0.14033373, 0.1019271, -0.01784891])],
                    [np.array([-0.11994868, 0.255285, -0.01709043])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.482, -0.876, -0.0, 0.0],
                                [-0.876, 0.482, 0.021, 0.0],
                                [-0.018, 0.01, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.01059847, -0.53727186, -0.84334254, 0.0],
                                [-0.01687089, 0.8431738, -0.53737634, 0.0],
                                [0.99980152, 0.01992333, -0.00012791, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.99956113, 0.02944234, -0.00326414, 0.0],
                                [0.02945254, -0.99956137, 0.0031231, 0.0],
                                [-0.00317076, -0.00321786, -0.99998987, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "leg2": {
                "name": "desk_leg2",
                "asset_file": "furniture/urdf/desk/desk_leg2.urdf",
                "ids": [117, 118, 119, 120],
                "reset_pos": [[-0.08, 0.07, -0.0175]] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "half_width": 0.0175,
                "default_assembled_pose": get_mat(
                    np.array([-0.080, 0.065625, 0.050001]), [0, -np.pi / 4, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([-0.035, 0.279, -0.015])],
                    [np.array([0.05981686, 0.10712636, -0.0224212])],
                    [np.array([0.18557446, 0.1461009, -0.02011154])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.888, 0.459, 0.014, 0.0],
                                [-0.459, -0.888, -0.01, 0.0],
                                [0.008, -0.015, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.18335986, 0.98291534, 0.01602139, 0.0],
                                [-0.98289353, 0.18359402, -0.01461495, 0.0],
                                [-0.01730669, -0.01306752, 0.99976486, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.01125821, 0.0096676, 0.99988991, 0.0],
                                [0.00722122, 0.999928, -0.00958666, 0.0],
                                [-0.99991053, 0.0071125, -0.01132721, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "leg3": {
                "name": "desk_leg3",
                "asset_file": "furniture/urdf/desk/desk_leg3.urdf",
                "ids": [121, 122, 123, 124],
                "reset_pos": [[0.08, 0.07, -0.0175]] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "half_width": 0.0175,
                "default_assembled_pose": get_mat(
                    np.array([0.080, 0.065625, -0.050001]), [0, -np.pi / 4, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.068, 0.323, -0.011])],
                    [np.array([0.1628194, 0.1732633, -0.02252014])],
                    [np.array([-0.17351371, 0.12436365, -0.01877135])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.343, -0.934, 0.097, 0.0],
                                [-0.909, 0.356, 0.217, 0.0],
                                [-0.237, -0.013, -0.971, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.80881345, -0.58802843, -0.00660876, 0.0],
                                [-0.58803391, 0.80883527, -0.00128764, 0.0],
                                [0.00610256, 0.00284471, -0.99997747, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.8064419, -0.5913133, -0.00050146, 0.0],
                                [-0.59120387, 0.80627567, 0.01993935, 0.0],
                                [-0.01138609, 0.01637639, -0.99980122, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "leg4": {
                "name": "desk_leg4",
                "asset_file": "furniture/urdf/desk/desk_leg4.urdf",
                "ids": [125, 126, 127, 128],
                "reset_pos": [[0.15, 0.07, -0.0175]] * 4 + [[0.14, 0.34, -0.095]],
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 4
                + [rot_mat([-np.pi / 2, 0, 0], hom=True)],
                "half_width": 0.0175,
                "default_assembled_pose": get_mat(
                    np.array([0.080, 0.065625, 0.050001]), [0, -np.pi / 4, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.166, 0.147, -0.02])],
                    [np.array([0.12731045, 0.3074856, -0.01941558])],
                    [np.array([0.0530014, 0.3194076, -0.02061281])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.011, -0.199, 0.98, 0.0],
                                [0.019, 0.98, 0.2, 0.0],
                                [-1.0, 0.021, -0.007, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.97035968, -0.24163274, 0.00395575, 0.0],
                                [0.24165221, 0.97034711, -0.00554586, 0.0],
                                [-0.00249839, 0.00633739, 0.99997681, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.00796196, -0.99176419, 0.12782967, 0.0],
                                [0.00122496, 0.12782392, 0.99179614, 0.0],
                                [-0.99996758, 0.00805318, 0.00019711, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
        "round_table": {
            "tag_size": 0.0195,
            "total_reward": 2.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.54, -0.05, 0.005], dtype=np.float32),
                np.array([0.553, 0.034, 0.031], dtype=np.float32),
                np.array([0.452, 0.096, 0.001], dtype=np.float32),
                np.array([0.565, 0.061, 0.08097788], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.85031855, -0.52503395, 0.02596223, 0.02497494], dtype=np.float32
                ),
                np.array(
                    [-0.8487733, -0.528669, -0.00558961, 0.00785169], dtype=np.float32
                ),
                np.array(
                    [-0.9152878, 0.01901048, 0.4020678, 0.01511118], dtype=np.float32
                ),
                np.array(
                    [-0.9793014, -0.08953558, -0.18145438, 0.00514412], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "round_table_top": {
                "name": "round_table_top",
                "asset_file": "furniture/urdf/round_table/round_table_top.urdf",
                "ids": [24, 25, 26, 27, 28, 29, 30, 31],
                "reset_pos": [[0, 0.24, -0.001]] * 2 + [[0.07, 0.26, -0.001]] * 3,
                "reset_ori": [rot_mat(np.array([0, np.pi, 0]), hom=True)] * 5,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([0.047, 0.216, -0.006])],
                    [np.array([-0.077, 0.19, -0.006])],
                    [np.array([0.076, 0.124, -0.003])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.986, 0.165, -0.004, 0.0],
                                [0.165, 0.986, 0.019, 0.0],
                                [0.007, 0.019, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.709, 0.705, -0.011, 0.0],
                                [0.705, 0.709, 0.008, 0.0],
                                [0.014, -0.002, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.956, 0.292, 0.002, 0.0],
                                [0.292, -0.956, 0.041, 0.0],
                                [0.013, -0.038, -0.999, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "round_table_leg": {
                "name": "round_table_leg",
                "asset_file": "furniture/urdf/round_table/round_table_leg.urdf",
                "ids": [32, 33, 34, 35],
                "reset_pos": [[0.13, 0.10, -0.02125]] * 4 + [[0.07, 0.26, -0.056]],
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 4
                + [rot_mat(np.array([-np.pi / 2, 0, 0]), hom=True)],
                "default_assembled_pose": get_mat(
                    # [0, 0, 0.044375], [np.pi / 2, 0, np.pi + -np.pi / 4]
                    [0, 0, 0.044375],
                    [np.pi / 2, 0, np.pi + np.pi / 36],
                ),
                "high_rand_reset_pos": [
                    [np.array([-0.118, 0.263, -0.039])],
                    [np.array([0.066, 0.301, -0.02])],
                    [np.array([-0.09, 0.155, -0.018])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.79042137, 0.370952, 0.48747155, -0.11553141],
                                [0.3175943, 0.9286462, -0.19170311, 0.25544176],
                                [-0.52380127, 0.00329194, -0.85183406, -0.01850056],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.402, 0.901, 0.16, 0.0],
                                [-0.915, -0.393, -0.088, 0.0],
                                [-0.016, -0.182, 0.983, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.855, -0.507, 0.106, 0.0],
                                [-0.517, 0.85, -0.103, 0.0],
                                [-0.038, -0.143, -0.989, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "round_table_base": {
                "name": "round_table_base",
                "asset_file": "furniture/urdf/round_table/round_table_base.urdf",
                "ids": [40, 41, 42, 43, 44, 45, 46, 47],
                "reset_pos": [[-0.12, 0.12, -0.02875]] * 5,
                "reset_ori": [rot_mat(np.array([np.pi, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    [0, 0.053125, 0], [-np.pi / 2, np.pi / 2, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([-0.135, 0.109, -0.03])],
                    [np.array([0.124, 0.16, -0.034])],
                    [np.array([-0.007, 0.293, -0.029])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.911, -0.413, -0.006, 0.0],
                                [-0.412, -0.91, 0.035, 0.0],
                                [-0.02, -0.029, -0.999, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.688, -0.726, -0.006, 0.0],
                                [-0.725, -0.688, 0.022, 0.0],
                                [-0.02, -0.011, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.967, 0.256, -0.01, 0.0],
                                [0.256, -0.967, 0.015, 0.0],
                                [-0.006, -0.017, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
        "drawer": {
            "tag_size": 0.0195,
            "total_reward": 2.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.44, 0.11, 0.095], dtype=np.float32),
                np.array([0.6369, 0.088, 0.1], dtype=np.float32),
                np.array([0.55, -0.11, 0.0077], dtype=np.float32),
                np.array([0.60, -0.055, 0.0337], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.9995351, -0.02676353, 0.01025268, 0.01040202], dtype=np.float32
                ),
                np.array(
                    [-0.7194083, 0.6810621, 0.03583809, 0.13161217], dtype=np.float32
                ),
                np.array([-1.0, 0, 0, 0], dtype=np.float32),
                np.array(
                    [0.74125683, 0.669928, -0.03118616, 0.02760757], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "drawer_box": {
                "name": "drawer_box",
                "asset_file": "furniture/urdf/drawer/drawer_box.urdf",
                "ids": [48, 49, 50, 52],  # Note that 51 is missing.
                "reset_pos": [[0.08, 0.13, -0.04125]] * 2
                + [[0.11, 0.30, -0.04125]] * 3,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, np.pi]), hom=True)] * 2
                + [rot_mat(np.array([np.pi / 2, 0, -np.pi / 2]), hom=True)] * 3,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([-0.078, 0.159, -0.042])],
                    [np.array([-0.101, 0.168, -0.043])],
                    [np.array([0.103, 0.118, -0.046])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.928, -0.001, 0.374, 0.0],
                                [0.374, -0.005, 0.927, 0.0],
                                [0.001, 1.0, 0.005, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.633, 0.005, -0.774, 0.0],
                                [-0.774, -0.021, 0.632, 0.0],
                                [-0.013, 1.0, 0.017, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.991, 0.007, 0.131, 0.0],
                                [0.131, -0.015, 0.991, 0.0],
                                [0.009, 1.0, 0.014, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "drawer_container_top": {
                "name": "drawer_container_top",
                "asset_file": "furniture/urdf/drawer/drawer_container_top.urdf",
                "ids": [53, 54, 55, 56, 57],
                "reset_pos": [[-0.11, 0.12, -0.0255]] * 5,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat([0, -0.0345, 0.008], [0, 0, 0]),
                "high_rand_reset_pos": [
                    [np.array([-0.1, 0.293, -0.029])],
                    [np.array([0.069, 0.137, -0.027])],
                    [np.array([-0.038, 0.156, -0.028])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-1.0, 0.016, -0.016, 0.0],
                                [-0.016, 0.012, 1.0, 0.0],
                                [0.016, 1.0, -0.012, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.871, -0.0, 0.492, 0.0],
                                [0.492, -0.025, -0.87, 0.0],
                                [0.012, 1.0, -0.021, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.9, 0.006, -0.436, 0.0],
                                [-0.436, -0.023, 0.9, 0.0],
                                [-0.004, 1.0, 0.023, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "drawer_container_bottom": {
                "name": "drawer_container_bottom",
                "asset_file": "furniture/urdf/drawer/drawer_container_bottom.urdf",
                "ids": [58, 59, 60, 61, 62],
                "reset_pos": [[-0.11, 0.25, -0.0255]] * 4 + [[0.055, 0.30, -0.03075]],
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, 0]), hom=True)] * 4
                + [rot_mat([np.pi / 2, 0, -np.pi / 2], hom=True)],
                "default_assembled_pose": get_mat([0, 0.0105, 0.008], [0, 0, 0]),
                "high_rand_reset_pos": [
                    [np.array([0.086, 0.173, -0.029])],
                    [np.array([0.034, 0.261, -0.027])],
                    [np.array([-0.085, 0.28, -0.018])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.221, -0.011, 0.975, 0.0],
                                [0.975, -0.002, -0.221, 0.0],
                                [0.005, 1.0, 0.01, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.938, 0.008, -0.346, 0.0],
                                [-0.346, -0.016, 0.938, 0.0],
                                [0.002, 1.0, 0.018, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.9, 0.006, -0.436, 0.0],
                                [-0.436, -0.023, 0.9, 0.0],
                                [-0.004, 1.0, 0.023, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
        "chair": {
            "tag_size": 0.0195,
            "total_reward": 5.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.59, 0.028, 0.007], dtype=np.float32),
                np.array([0.60780913, 0.08148123, 0.07632343], dtype=np.float32),
                np.array([0.4668, 0.0572, 0.0147], dtype=np.float32),
                np.array([0.66, 0.133, 0.107], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.99883515, -0.02959231, -0.01156779, 0.03631632],
                    dtype=np.float32,
                ),
                np.array(
                    [-0.9989915, 0.00532738, -0.02000867, 0.0398413], dtype=np.float32
                ),
                np.array(
                    [-0.8763746, 0.06941535, 0.47324368, 0.05647525], dtype=np.float32
                ),
                np.array(
                    [-0.9711292, -0.07015788, -0.22556745, 0.03324548], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "chair_seat": {
                "name": "chair_seat",
                "asset_file": "furniture/urdf/chair/chair_seat.urdf",
                "ids": [79, 80, 81, 82],
                "reset_pos": [[0.03, 0.27, -0.015]] * 2 + [[0.115, 0.31, -0.015]] * 3,
                "reset_ori": [rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True)] * 5,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([0.05986764, 0.19544938, -0.01780899])],
                    [np.array([-0.17163497, 0.15262844, -0.01651937])],
                    [np.array([0.06676043, 0.24115437, -0.0212097])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.92918456, -0.01316611, -0.36938149, 0.0],
                                [0.3693251, 0.00657321, -0.929277, 0.0],
                                [0.014663, -0.9998917, -0.00124513, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.83291751, -0.006127, 0.55336332, 0.0],
                                [-0.55339634, 0.00741897, -0.83288503, 0.0],
                                [0.00099766, -0.99995375, -0.00957008, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.02653122, -0.01253995, 0.99956936, 0.0],
                                [-0.99961007, -0.0083738, -0.02663738, 0.0],
                                [0.00870425, -0.99988633, -0.01231295, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "chair_back": {
                "name": "chair_back",
                "asset_file": "furniture/urdf/chair/chair_back.urdf",
                "ids": [83, 84, 85, 86, 87, 88, 89, 90],
                "reset_pos": [[-0.10, 0.20, -0.015]] * 5,
                "reset_ori": [rot_mat(np.array([0, np.pi, 0]), hom=True)] * 5,
                "high_rand_reset_pos": [
                    [np.array([-0.08739488, 0.19979191, -0.01221029])],
                    [np.array([0.0152305, 0.17737612, -0.01473497])],
                    [np.array([-0.10485372, 0.15906905, -0.01489258])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.76807487, 0.64029282, -0.00926405, 0.0],
                                [0.64025527, 0.76813066, 0.00695575, 0.0],
                                [0.01156972, -0.00058881, -0.99993277, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.15836088, -0.98732817, -0.0102406, 0.0],
                                [-0.98735344, 0.15827014, 0.00914752, 0.0],
                                [-0.00741083, 0.0115597, -0.99990565, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.46160662, 0.8870849, -0.00000017, -0.10479736],
                                [0.8870849, 0.46160638, 0.00000086, 0.15731335],
                                [0.00000084, 0.00000025, -1.0000002, -0.01489258],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "chair_leg1": {
                "name": "chair_leg1",
                "asset_file": "furniture/urdf/chair/chair_leg1.urdf",
                "ids": [91, 92, 93, 94],
                "reset_pos": [[0.01, 0.12, -0.015]] * 5,
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    [0.03375, 0.045, -0.01875], [0, np.pi / 2, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.12987243, 0.1504644, -0.0237539])],
                    [np.array([-0.07327363, 0.28158466, -0.01686125])],
                    [np.array([0.09447257, 0.10305008, -0.02093053])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.01107043, 0.12764195, 0.99175853, 0.0],
                                [0.00335702, 0.99181849, -0.12761219, 0.0],
                                [-0.99993306, 0.00191663, -0.01140836, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.00929469, -0.8878895, 0.45996296, 0.0],
                                [0.00425327, 0.4599435, 0.88793802, 0.0],
                                [-0.99994779, 0.01020944, -0.00049862, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.9924311, -0.12201616, -0.01388437, 0.0],
                                [-0.12223718, 0.9923659, 0.01637016, 0.0],
                                [0.01178095, 0.01794344, -0.99976969, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "chair_leg2": {
                "name": "chair_leg2",
                "asset_file": "furniture/urdf/chair/chair_leg2.urdf",
                "ids": [95, 96, 97, 98],
                "reset_pos": [[0.07, 0.12, -0.015]] * 4
                + [[0.105 + 0.03375 + 0.01, 0.33 + 0.01875, -0.07359187]],
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 4
                + [rot_mat(np.array([-np.pi / 2, 0, 0]), hom=True)],
                "default_assembled_pose": get_mat(
                    [-0.03375, 0.045, -0.01875], [0, np.pi / 2, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.08744325, 0.32398197, -0.02274978])],
                    [np.array([0.02714946, 0.31022108, -0.01758601])],
                    [np.array([0.16358277, 0.11960384, -0.01879693])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.00169327, 0.30636734, -0.95191187, 0.0],
                                [-0.00564334, 0.95189512, 0.30637199, 0.0],
                                [0.99998266, 0.00589073, 0.00011711, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.99577844, -0.08591485, 0.03231045, 0.0],
                                [0.08579749, 0.99630004, 0.00500382, 0.0],
                                [-0.03262081, -0.00221054, 0.99946535, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.97426367, -0.22378716, 0.0270115, 0.0],
                                [0.22386295, 0.97462058, 0.00022389, 0.0],
                                [-0.02637606, 0.00582875, 0.9996351, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "chair_nut1": {
                "name": "chair_nut1",
                "asset_file": "furniture/urdf/chair/chair_nut1.urdf",
                "ids": [99, 100, 101, 102, 103],
                "reset_pos": [[0.13, 0.12, -0.015]] * 5,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    [-0.035, 0, 0.0815], [-np.pi / 2, 0, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.01052985, 0.30740196, -0.02801788])],
                    [np.array([0.0829217, 0.27938807, -0.03095993])],
                    [np.array([-0.05123203, 0.29473808, -0.02397889])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.05493468, 0.02182969, 0.99825132, 0.0],
                                [0.99848986, -0.00077346, -0.05493096, 0.0],
                                [-0.0004271, 0.99976146, -0.02183929, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.16628975, 0.02535662, 0.98575091, 0.0],
                                [0.98604178, -0.01272193, -0.16601163, 0.0],
                                [0.00833112, 0.99959761, -0.02711824, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.95705044, 0.00778796, 0.2898168, 0.0],
                                [0.28989899, -0.01327884, -0.95696509, 0.0],
                                [-0.00360438, 0.99988151, -0.01496624, 0.0],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                ],
                            ]
                        )
                    ],
                ],
            },
            "chair_nut2": {
                "name": "chair_nut2",
                "asset_file": "furniture/urdf/chair/chair_nut2.urdf",
                "ids": [104, 105, 106, 107, 108],
                "reset_pos": [[0.19, 0.12, -0.015]] * 5,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    [0.035, 0, 0.0815], [-np.pi / 2, 0, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([-0.12835887, 0.289353, -0.02523068])],
                    [np.array([-0.13410832, 0.26472944, -0.02297941])],
                    [np.array([-0.13044557, 0.29677352, -0.02132733])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.18104237, 0.00664857, 0.98345286, 0.0],
                                [0.98347282, -0.00349244, -0.18102252, 0.0],
                                [0.00223106, 0.99997187, -0.00717101, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.50522768, 0.01058421, 0.86292118, 0.0],
                                [0.86298418, -0.00828896, -0.50516284, 0.0],
                                [0.00180602, 0.99990964, -0.0133218, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.2258541, 0.01629639, 0.97402483, 0.0],
                                [0.97405064, -0.01128122, 0.22604883, 0.0],
                                [0.01467198, 0.99980354, -0.0133256, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
        "lamp": {
            "tag_size": 0.0195,
            "total_reward": 2.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.46, 0.010, 0.054], dtype=np.float32),
                np.array([0.62, 0.11, 0.05768], dtype=np.float32),
                np.array([0.5, 0.144, 0.012], dtype=np.float32),
                np.array([0.6299, 0.1088, 0.14063806], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array([-1.0, 0, 0, 0], dtype=np.float32),
                np.array(
                    [-0.998632, 0.01359205, -0.01235274, 0.04895734], dtype=np.float32
                ),
                np.array(
                    [0.9007951, -0.0191828, -0.43380633, 0.0035078], dtype=np.float32
                ),
                np.array(
                    [-0.95469904, -0.10741911, -0.27635896, 0.02523001],
                    dtype=np.float32,
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "lamp_hood": {
                "name": "lamp_hood",
                "asset_file": "furniture/urdf/lamp/lamp_hood.urdf",
                "ids": [163, 164, 165, 166, 167, 168],
                "reset_pos": [[-0.12, 0.16, -0.048583]] * 5,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, 0]), hom=True)] * 5,
                "high_rand_reset_pos": [
                    [np.array([0.084, 0.141, -0.053])],
                    [np.array([-0.088, 0.12, -0.05])],
                    [np.array([0.081, 0.284, -0.052])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.995, 0.014, -0.095, 0.0],
                                [-0.095, -0.01, 0.995, 0.0],
                                [0.013, 1.0, 0.011, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.971, 0.046, 0.235, 0.0],
                                [0.238, 0.063, 0.969, 0.0],
                                [0.03, 0.997, -0.072, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.839, 0.005, 0.544, 0.0],
                                [0.544, -0.015, 0.839, 0.0],
                                [0.012, 1.0, 0.01, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "lamp_base": {
                "name": "lamp_base",
                "asset_file": "furniture/urdf/lamp/lamp_base.urdf",
                "ids": [169, 170, 171, 172, 173],
                "reset_pos": [[0.05, 0.14, -0.02]] * 2 + [[0.12, 0.31, -0.02]] * 3,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, 0]), hom=True)] * 5,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([-0.129, 0.205, -0.02])],
                    [np.array([-0.03, 0.26, -0.019])],
                    [np.array([-0.072, 0.248, -0.019])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.911, 0.005, -0.413, 0.0],
                                [-0.413, -0.028, -0.91, 0.0],
                                [-0.016, 1.0, -0.023, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.824, 0.005, 0.567, 0.0],
                                [0.567, -0.01, -0.824, 0.0],
                                [0.001, 1.0, -0.011, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.737, 0.004, 0.675, 0.0],
                                [0.675, -0.01, -0.737, 0.0],
                                [0.004, 1.0, -0.01, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "lamp_bulb": {
                "name": "lamp_bulb",
                "asset_file": "furniture/urdf/lamp/lamp_bulb.urdf",
                "ids": [174, 177, 176, 175],
                "reset_pos": [[0.18, 0.13, -0.03]] * 4 + [[0.12, 0.31, -0.09]],
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 4
                + [rot_mat([-np.pi / 2, 0, 0], hom=True)],
                "high_rand_reset_pos": [
                    [np.array([0.07429203, 0.29051307, -0.0336427])],
                    [np.array([0.077, 0.156, -0.023])],
                    [np.array([0.03684583, 0.15773909, -0.0237988])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.0242578, -0.98711663, -0.15815291, 0.0],
                                [0.93183929, -0.07962076, 0.35402828, 0.0],
                                [-0.36205944, -0.13878517, 0.92176551, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.967, -0.143, 0.211, 0.0],
                                [0.116, 0.984, 0.135, 0.0],
                                [-0.227, -0.106, 0.968, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.60462284, -0.72320396, -0.33377719, 0.0],
                                [0.7204029, 0.67527962, -0.15816779, 0.0],
                                [0.33978051, -0.14482222, 0.92928773, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
        "stool": {
            "tag_size": 0.0195,
            "total_reward": 3.0,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.5695, -0.0104, 0.018], dtype=np.float32),
                np.array([0.5816, 0.1044, 0.0448], dtype=np.float32),
                np.array([0.5019, 0.08, 0.01], dtype=np.float32),
                np.array([0.61854601, 0.10544886, 0.12530436], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.86606866, -0.4986022, 0.02819192, 0.02293851], dtype=np.float32
                ),
                np.array(
                    [-0.88500136, -0.46518293, 0.00696717, 0.01813608], dtype=np.float32
                ),
                np.array(
                    [-0.86738664, -0.00651512, 0.4972701, 0.01789967], dtype=np.float32
                ),
                np.array(
                    [-0.9632059, -0.0273593, -0.26139688, 0.05619244], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "stool_seat": {
                "name": "stool_seat",
                "asset_file": "furniture/urdf/stool/stool_seat.urdf",
                "ids": [143, 144, 145, 146],
                "reset_pos": [[0, 0.29, -0.015]] * 2 + [[0.11, 0.30, -0.015]] * 3,
                "reset_ori": [rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True)] * 5,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([-0.06, 0.227, -0.016])],
                    [np.array([0.027, 0.249, -0.016])],
                    [np.array([0.13124122, 0.09935549, -0.02212791])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.876, 0.247, -0.414, 0.0],
                                [0.474, 0.292, -0.83, 0.0],
                                [-0.084, -0.924, -0.373, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.174, 0.003, -0.985, 0.0],
                                [0.984, 0.028, -0.174, 0.0],
                                [0.027, -1.0, -0.008, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.754682, -0.01281741, -0.65596557, 0.0],
                                [0.65574956, 0.01750702, -0.75477552, 0.0],
                                [0.02115828, -0.99976456, -0.00480723, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "stool_leg1": {
                "name": "stool_leg1",
                "asset_file": "furniture/urdf/stool/stool_leg1.urdf",
                "ids": [147, 148, 149, 150],
                "reset_pos": [[-0.10, 0.18, -0.0175]] * 5,
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    [0, 0.045, 0.040542], [0, np.pi / 2, 0]
                ),
                "high_rand_reset_pos": [
                    [np.array([0.076, 0.287, -0.021])],
                    [np.array([0.11, 0.151, -0.02])],
                    [np.array([-0.11369069, 0.19002846, -0.01333283])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.353, -0.936, 0.01, 0.0],
                                [0.936, 0.353, -0.005, 0.0],
                                [0.001, 0.011, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.001, -0.491, -0.871, 0.0],
                                [-0.022, 0.871, -0.491, 0.0],
                                [1.0, 0.02, -0.01, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.01497109, -0.00862187, 0.99985069, 0.0],
                                [-0.24469948, 0.96958756, 0.00469694, 0.0],
                                [-0.96948332, -0.24459264, -0.01662555, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "stool_leg2": {
                "name": "stool_leg2",
                "asset_file": "furniture/urdf/stool/stool_leg2.urdf",
                "ids": [151, 152, 153, 154],
                "reset_pos": [[0, 0.18, -0.0175]] * 5,
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat([0, 0, 0], [0, 2 * np.pi / 3, 0])
                @ get_mat([0, 0.045, 0.040542], [0, np.pi / 2, 0]),
                "high_rand_reset_pos": [
                    [np.array([-0.134, 0.186, -0.02])],
                    [np.array([-0.045, 0.188, -0.018])],
                    [np.array([0.06150678, 0.28508818, -0.02003413])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.083, -0.705, -0.705, 0.0],
                                [-0.104, 0.697, -0.709, 0.0],
                                [0.991, 0.132, -0.015, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.027, -0.122, -0.992, 0.0],
                                [0.239, 0.963, -0.125, 0.0],
                                [0.971, -0.24, 0.003, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.03810083, 0.09744918, 0.99451101, 0.0],
                                [-0.25105661, 0.96237803, -0.10391883, 0.0],
                                [-0.96722233, -0.25363794, -0.01220215, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "stool_leg3": {
                "name": "stool_leg3",
                "asset_file": "furniture/urdf/stool/stool_leg3.urdf",
                "ids": [155, 156, 157, 158],
                "reset_pos": [[0.10, 0.18, -0.0175]] * 4
                + [[0.11 + 0.0351104, 0.30 + 0.020271, -0.07]],
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 4
                + [rot_mat([-np.pi / 2, 0, np.pi], hom=True)],
                "default_assembled_pose": get_mat([0, 0, 0], [0, 4 * np.pi / 3, 0])
                @ get_mat([0, 0.045, 0.040542], [0, np.pi / 2, 0]),
                "high_rand_reset_pos": [
                    [np.array([0.126, 0.171, -0.02])],
                    [np.array([-0.108, 0.253, -0.02])],
                    [np.array([0.02115019, 0.18929027, -0.0199405])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.008, -0.101, -0.995, 0.0],
                                [-0.024, 0.995, -0.101, 0.0],
                                [1.0, 0.024, 0.005, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.033, 0.242, 0.97, 0.0],
                                [-0.144, -0.959, 0.244, 0.0],
                                [0.989, -0.148, 0.004, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.87092847, -0.4912734, 0.0115781, 0.0],
                                [
                                    0.49131191,
                                    0.87098348,
                                    -0.00056219,
                                    0.0,
                                ],
                                [
                                    -0.00980814,
                                    0.00617809,
                                    0.99993283,
                                    0.0,
                                ],
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                ],
                            ]
                        )
                    ],
                ],
            },
        },
        "cabinet": {
            "tag_size": 0.0195,
            "total_reward": 3.0,
            "total_phase": 11,
            "ee_pos": [
                np.array(
                    [0.56077659, 0.04639731, 0.11256177], dtype=np.float32
                ),  # Dummy for offset
                np.array([0.451, 0.0275 * 2 - 0.01, 0.025], dtype=np.float32),
                np.array([0.503, 0.119, 0.060], dtype=np.float32),
                np.array([0.41, 0.20, 0.015], dtype=np.float32),
                np.array([0.573, 0.128, 0.08], dtype=np.float32),
            ],
            "ee_quat": [
                np.array(
                    [-8.9071459e-01, -4.5411372e-01, 2.0201905e-02, 3.7367147e-04],
                    dtype=np.float32,
                ),  # Dummy for offset
                np.array(
                    [-0.72498286, -0.68866396, -0.00298789, 0.01152807],
                    dtype=np.float32,
                ),
                np.array(
                    [-0.7326354, -0.68018675, -0.00747206, 0.02314179], dtype=np.float32
                ),
                np.array(
                    [-0.7590536, -0.64950174, -0.01921348, 0.04019891], dtype=np.float32
                ),
                np.array(
                    [-0.7481479, -0.66248906, -0.02970546, 0.02237379], dtype=np.float32
                ),
            ],
            "grippers": [-1, 1, -1, 1, -1],
            "cabinet_body": {
                "name": "cabinet_body",
                "asset_file": "furniture/urdf/cabinet/cabinet_body.urdf",
                "ids": [134, 135, 136, 137, 138],
                # 'reset_pos': [[0.16, 0.13, -0.02875]],[[-0.04, 0.09, -0.01125]]
                "reset_pos": [[0, 0.24, -0.02875]] * 2 + [[0.10, 0.28, -0.02875]] * 3,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "part_moved_skill_idx": 2,
                "high_rand_reset_pos": [
                    [np.array([0.105, 0.128, -0.034])],
                    [np.array([0.076, 0.273, -0.033])],
                    [np.array([-0.08, 0.265, -0.03])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.016, 0.791, 0.612, 0.0],
                                [-0.011, -0.611, 0.791, 0.0],
                                [1.0, -0.019, -0.001, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.018, 0.953, -0.301, 0.0],
                                [-0.004, 0.301, 0.954, 0.0],
                                [1.0, -0.016, 0.009, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.013, -0.156, -0.988, 0.0],
                                [0.002, 0.988, -0.156, 0.0],
                                [1.0, -0.0, 0.013, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "cabinet_door_left": {
                "name": "cabinet_door_left",
                "asset_file": "furniture/urdf/cabinet/cabinet_door_left.urdf",
                "ids": [139, 140],
                "reset_pos": [[0.12, 0.08, -0.01125]] * 5,
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 5,
                "high_rand_reset_pos": [
                    [np.array([0.045, 0.221, -0.01])],
                    [np.array([-0.056, 0.315, -0.003])],
                    [np.array([0.076, 0.317, -0.011])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.79, 0.613, 0.009, 0.0],
                                [-0.613, 0.79, -0.009, 0.0],
                                [-0.013, 0.001, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.823, -0.569, 0.007, 0.0],
                                [0.568, -0.823, -0.018, 0.0],
                                [0.016, -0.011, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.453, -0.892, 0.006, 0.0],
                                [0.891, -0.453, -0.012, 0.0],
                                [0.014, -0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "cabinet_door_right": {
                "name": "cabinet_door_right",
                "asset_file": "furniture/urdf/cabinet/cabinet_door_right.urdf",
                "ids": [141, 142],
                "reset_pos": [[0.20, 0.08, -0.01125]] * 4 + [[0.125, 0.2425, -0.05625]],
                "reset_ori": [rot_mat(np.array([0, 0, 0]), hom=True)] * 5,
                "high_rand_reset_pos": [
                    [np.array([-0.077, 0.139, -0.011])],
                    [np.array([0.077, 0.081, -0.012])],
                    [np.array([0.133, 0.199, -0.013])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [0.564, -0.826, 0.001, 0.0],
                                [0.826, 0.563, -0.028, 0.0],
                                [0.023, 0.017, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.861, -0.508, -0.003, 0.0],
                                [0.508, 0.861, -0.027, 0.0],
                                [0.016, 0.022, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [0.962, -0.271, 0.019, 0.0],
                                [0.272, 0.962, -0.013, 0.0],
                                [-0.015, 0.018, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
            "cabinet_top": {
                "name": "cabinet_top",
                "asset_file": "furniture/urdf/cabinet/cabinet_top.urdf",
                "ids": [129, 130, 131, 132, 133],
                "reset_pos": [[-0.15, 0.15, -0.015]] * 5,
                "reset_ori": [rot_mat(np.array([np.pi / 2, 0, np.pi / 2]), hom=True)]
                * 5,
                "high_rand_reset_pos": [
                    [np.array([-0.068, 0.303, -0.015])],
                    [np.array([-0.061, 0.169, -0.016])],
                    [np.array([0.065, 0.135, -0.017])],
                ],
                "high_rand_reset_ori": [
                    [
                        np.array(
                            [
                                [-0.993, 0.003, 0.117, 0.0],
                                [0.117, -0.02, 0.993, 0.0],
                                [0.006, 1.0, 0.02, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.854, 0.004, -0.521, 0.0],
                                [-0.521, -0.008, 0.854, 0.0],
                                [-0.001, 1.0, 0.009, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                    [
                        np.array(
                            [
                                [-0.962, 0.005, 0.273, 0.0],
                                [0.273, -0.002, 0.962, 0.0],
                                [0.005, 1.0, 0.001, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    ],
                ],
            },
        },
    },
}
