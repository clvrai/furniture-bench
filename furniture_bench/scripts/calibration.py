"""Calibrate extrinsic of the front camera"""
import argparse
import time
import cv2
from pathlib import Path

import numpy as np

from furniture_bench.utils.pose import mat_to_roll_pitch_yaw
from furniture_bench.perception.apriltag import AprilTag
from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.utils.detection import get_cam_to_base
from furniture_bench.config import config
from furniture_bench.utils.draw import draw_axis


ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")


avg_pose = {
    "desk": np.array(
        [
            [0.99999279, 0.00160516, -0.00344911, 0.01415162],
            [-0.00378206, 0.32150161, -0.9469015, 0.77910175],
            [-0.00041103, 0.9469077, 0.32150534, -0.25672432],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "one_leg": np.array(
        [
            [0.99970633, -0.0127885, -0.02058551, 0.03352087],
            [-0.01602097, 0.28858966, -0.95731884, 0.79602363],
            [0.01818344, 0.95736748, 0.28830001, -0.28269542],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "round_table": np.array(
        [
            [0.99982584, 0.01780559, -0.00558734, 0.02009043],
            [-0.01143877, 0.34817287, -0.93736058, 0.75832786],
            [-0.0147449, 0.93726128, 0.34831589, -0.25012811],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "cabinet": np.array(
        [
            [0.99966449, 0.01082194, 0.02353297, -0.01055085],
            [0.01951828, 0.28254554, -0.9590553, 0.81339255],
            [-0.01702798, 0.95919287, 0.28223953, -0.23436418],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "chair": np.array(
        [
            [0.99985361, 0.01687093, 0.0028574, 0.02373987],
            [-0.00329247, 0.3535561, -0.93540758, 0.7633147],
            [-0.01679145, 0.93526119, 0.35355988, -0.25721807],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "stool": np.array(
        [
            [0.99995387, -0.00010306, -0.00960495, 0.0266552],
            [-0.00894408, 0.35464668, -0.93495762, 0.76644606],
            [0.00350272, 0.93500042, 0.3546294, -0.25921994],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "lamp": np.array(
        [
            [0.99957258, 0.0237143, 0.01709732, 0.01008731],
            [0.00677436, 0.38102335, -0.92454064, 0.75995131],
            [-0.02843931, 0.92426133, 0.38069984, -0.26012454],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "square_table": np.array(
        [
            [0.99942148, -0.00410567, 0.03376222, 0.00306158],
            [0.03301501, 0.35558149, -0.934062, 0.79865629],
            [-0.00817027, 0.93463624, 0.35551131, -0.25192554],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "drawer": np.array(
        [
            [0.99866104, 0.00004919, 0.05173153, 0.00445381],
            [0.04791539, 0.37607616, -0.925349, 0.80655273],
            [-0.0195005, 0.92658877, 0.37557024, -0.24571178],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

avg_pose_mixed = {
    "desk": np.array(
        [
            [0.99987799, 0.01562193, 0.00005377, 0.01141237],
            [-0.00530092, 0.34251836, -0.93949622, 0.76871677],
            [-0.01469516, 0.9393813, 0.34255937, -0.25737663],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "drawer": np.array(
        [
            [0.99863744, 0.00085928, 0.05217813, -0.00059859],
            [0.0481806, 0.36893785, -0.92820448, 0.80643984],
            [-0.02004807, 0.92945367, 0.36839375, -0.24205452],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "lamp": np.array(
        [
            [0.99937046, 0.02121121, 0.02843958, 0.00280873],
            [0.01888875, 0.36044857, -0.9325878, 0.7688803],
            [-0.03003232, 0.93253785, 0.35982099, -0.25412462],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "round_table": np.array(
        [
            [0.99994165, 0.01066213, -0.00173869, 0.01478042],
            [-0.00505905, 0.31996414, -0.94741613, 0.7652581],
            [-0.00954515, 0.94736964, 0.31999943, -0.24948219],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "square_table": np.array(
        [
            [0.99942493, -0.00336128, 0.03374103, 0.00100822],
            [0.03278368, 0.34990945, -0.93620968, 0.79982083],
            [-0.00865944, 0.93677747, 0.34981844, -0.25115504],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "cabinet": np.array(
        [
            [0.99892062, 0.01174908, 0.04493921, -0.00428806],
            [0.03913965, 0.30808488, -0.95055342, 0.80157242],
            [-0.02501322, 0.95128632, 0.30729249, -0.23691384],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "chair": np.array(
        [
            [0.99986511, 0.01338788, 0.00951535, 0.01392921],
            [0.00396246, 0.36560041, -0.93076348, 0.76792676],
            [-0.01593977, 0.93067563, 0.36549804, -0.26082106],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "stool": np.array(
        [
            [0.99998289, 0.00361912, 0.0046001, 0.01837403],
            [0.00319156, 0.32165495, -0.94685155, 0.77645035],
            [-0.00490642, 0.94685, 0.32163787, -0.25567752],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

avg_pose["setup_front"] = avg_pose["one_leg"]
avg_pose["obstacle"] = avg_pose["one_leg"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="one_leg")

    args = parser.parse_args()
    cam2 = RealsenseCam(
        config["camera"][2]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
        None,
        disable_auto_exposure=True,
    )

    april_tag = AprilTag(config["furniture"]["base_tag_size"])
    prev_img = cv2.imread(f"{ASSET_ROOT}/calibration/{args.target}.png")
    while True:
        color_img, _ = cam2.get_image()
        cam2_to_base = get_cam_to_base(cam2, 2, april_tag=april_tag)
        dst = cv2.addWeighted(color_img, 0.5, prev_img, 0.3, 0)

        if cam2_to_base is None:
            cv2.imshow(
                "calibration",
                cv2.cvtColor(cv2.resize(dst, (1280, 720)), cv2.COLOR_BGR2RGB),
            )
            time.sleep(0.001)

            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
            continue

        pos_error_threshold = 0.004  # 4mm
        rot_error_threshold = 1.0  # 1.0 degree

        cv2.putText(
            dst,
            "x pos: {0}{1:0.3f}".format(
                "+" if cam2_to_base[0, 3] - avg_pose[args.target][0, 3] > 0 else "",
                (cam2_to_base[0, 3] - avg_pose[args.target][0, 3]),
            ),
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            thickness=4,
            color=(
                (255, 0, 0)
                if abs(cam2_to_base[0, 3] - avg_pose[args.target][0, 3])
                > pos_error_threshold
                else (0, 255, 0)
            ),
        )

        cv2.putText(
            dst,
            "y pos: {0}{1:0.3f}".format(
                "+" if cam2_to_base[1, 3] - avg_pose[args.target][1, 3] > 0 else "",
                (cam2_to_base[1, 3] - avg_pose[args.target][1, 3]),
            ),
            org=(50, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            thickness=4,
            color=(
                (255, 0, 0)
                if abs(cam2_to_base[1, 3] - avg_pose[args.target][1, 3])
                > pos_error_threshold
                else (0, 255, 0)
            ),
        )

        cv2.putText(
            dst,
            "z pos: {0}{1:0.3f}".format(
                "+" if cam2_to_base[2, 3] - avg_pose[args.target][2, 3] > 0 else "",
                (cam2_to_base[2, 3] - avg_pose[args.target][2, 3]),
            ),
            org=(50, 150),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            thickness=4,
            color=(
                (255, 0, 0)
                if abs(cam2_to_base[2, 3] - avg_pose[args.target][2, 3])
                > pos_error_threshold
                else (0, 255, 0)
            ),
        )

        cam2_xyz_rot = mat_to_roll_pitch_yaw(cam2_to_base)
        avg_pose_rot = mat_to_roll_pitch_yaw(avg_pose[args.target])

        cv2.putText(
            dst,
            "x rot: {0}{1:0.3f}".format(
                "+"
                if (np.degrees(cam2_xyz_rot[0]) - np.degrees(avg_pose_rot[0])) > 0
                else "",
                (np.degrees(cam2_xyz_rot[0]) - np.degrees(avg_pose_rot[0])),
            ),
            org=(50, 200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            thickness=4,
            color=(
                (255, 0, 0)
                if abs(np.degrees(cam2_xyz_rot[0]) - np.degrees(avg_pose_rot[0]))
                > rot_error_threshold
                else (0, 255, 0)
            ),
        )
        cv2.putText(
            dst,
            "y rot: {0}{1:0.3f}".format(
                "+"
                if (np.degrees(cam2_xyz_rot[1]) - np.degrees(avg_pose_rot[1])) > 0
                else "",
                (np.degrees(cam2_xyz_rot[1]) - np.degrees(avg_pose_rot[1])),
            ),
            org=(50, 250),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            thickness=4,
            color=(
                (255, 0, 0)
                if abs(np.degrees(cam2_xyz_rot[1]) - np.degrees(avg_pose_rot[1]))
                > rot_error_threshold
                else (0, 255, 0)
            ),
        )
        cv2.putText(
            dst,
            "z rot: {0}{1:0.3f}".format(
                "+"
                if (np.degrees(cam2_xyz_rot[2]) - np.degrees(avg_pose_rot[2])) > 0
                else "",
                (np.degrees(cam2_xyz_rot[2]) - np.degrees(avg_pose_rot[2])),
            ),
            org=(50, 300),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            thickness=4,
            color=(
                (255, 0, 0)
                if abs(np.degrees(cam2_xyz_rot[2]) - np.degrees(avg_pose_rot[2]))
                > rot_error_threshold
                else (0, 255, 0)
            ),
        )

        cam_intr = cam2.intr_mat
        base_pose = np.linalg.inv(cam2_to_base) @ np.eye(4)  # Base tag.
        dst = draw_axis(
            dst,
            base_pose[:3, :3],
            base_pose[:3, 3],
            cam_intr,
            0.05,
            5,
            text_label=True,
            draw_arrow=True,
        )

        cv2.imshow(
            "calibration", cv2.cvtColor(cv2.resize(dst, (1280, 720)), cv2.COLOR_BGR2RGB)
        )

        time.sleep(0.001)

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
