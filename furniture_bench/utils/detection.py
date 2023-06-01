import time
from datetime import datetime
from multiprocessing import shared_memory

import numpy as np
from numpy.linalg import inv
import cv2

import furniture_bench.utils.transform as T
from furniture_bench.utils.frequency import set_frequency
from furniture_bench.perception.apriltag import AprilTag
from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.utils.pose import comp_avg_pose
from furniture_bench.perception.realsense import read_detect
from furniture_bench.config import config


def get_cam_to_base(cam=None, cam_num=-1, img=None, cam_intr=None, april_tag=None):
    """Get homogeneous transforms that maps camera points to base points."""
    if april_tag is None:
        april_tag = AprilTag(config["furniture"]["base_tag_size"])
    if img is None:
        color_frame, _ = cam.get_frame()
    else:
        color_frame = img

    if cam_intr is not None:
        intr = cam_intr
    else:
        intr = cam.intr_param

    tags = april_tag.detect_id(color_frame, intr)

    trials = 10
    cam_to_bases = []
    for _ in range(trials):
        for base in config["furniture"]["base_tags"]:
            base_tag = tags.get(base)
            if base_tag is not None:
                rel_pose = config["furniture"]["rel_pose_from_coordinate"][base]
                transform = T.to_homogeneous(
                    base_tag.pose_t, base_tag.pose_R
                ) @ np.linalg.inv(rel_pose)
                cam_to_bases.append(np.linalg.inv(transform))

        tags = april_tag.detect_id(color_frame, intr)
        time.sleep(0.01)

    #  All the base tags are not detected.
    if len(cam_to_bases) == 0:
        return None
        # raise Exception(f"camera {cam_num}: Base tags are not detected.")
    cam_to_base = comp_avg_pose(cam_to_bases)
    return cam_to_base


def detect_bases(cam):
    april_tag = AprilTag(config["furniture"]["base_tag_size"])
    color_frame, _ = cam.get_frame()
    tags = april_tag.detect_id(color_frame, cam.intr_param)

    bases = []
    for base in config["furniture"]["base_tags"]:
        base_tag = tags.get(base)
        bases.append(base_tag)

    return bases


def detection_loop(config, parts, num_parts, tag_size, lock, shm):
    print("Start detection")
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
        config["camera"][2]["roi"],
    )
    cam3 = RealsenseCam(
        config["camera"][3]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
    )
    print("Camera initialized")

    cam1_to_base = None
    cam2_to_base = get_cam_to_base(cam2, 2)
    cam3_to_base = get_cam_to_base(cam3, 3)

    april_tag = AprilTag(tag_size)
    color_shape = (
        config["camera"]["color_img_size"][1],
        config["camera"]["color_img_size"][0],
        3,
    )
    depth_shape = (
        config["camera"]["depth_img_size"][1],
        config["camera"]["depth_img_size"][0],
    )

    while True:
        detection = _get_parts_poses(
            parts,
            num_parts,
            april_tag,
            cam1,
            cam2,
            cam3,
            cam1_to_base,
            cam2_to_base,
            cam3_to_base,
        )
        parts_poses_shm = shared_memory.SharedMemory(name=shm[0])
        parts_founds_shm = shared_memory.SharedMemory(name=shm[1])
        color_shm1 = shared_memory.SharedMemory(name=shm[2])
        depth_shm1 = shared_memory.SharedMemory(name=shm[3])
        color_shm2 = shared_memory.SharedMemory(name=shm[4])
        depth_shm2 = shared_memory.SharedMemory(name=shm[5])
        color_shm3 = shared_memory.SharedMemory(name=shm[6])
        depth_shm3 = shared_memory.SharedMemory(name=shm[7])

        parts_poses = np.ndarray(
            shape=(num_parts * 7,), dtype=np.float32, buffer=parts_poses_shm.buf
        )
        parts_found = np.ndarray(
            shape=(num_parts,), dtype=bool, buffer=parts_founds_shm.buf
        )
        color_img1 = np.ndarray(
            shape=color_shape, dtype=np.uint8, buffer=color_shm1.buf
        )
        depth_img1 = np.ndarray(
            shape=depth_shape, dtype=np.uint16, buffer=depth_shm1.buf
        )
        color_img2 = np.ndarray(
            shape=color_shape, dtype=np.uint8, buffer=color_shm2.buf
        )
        depth_img2 = np.ndarray(
            shape=depth_shape, dtype=np.uint16, buffer=depth_shm2.buf
        )
        color_img3 = np.ndarray(
            shape=color_shape, dtype=np.uint8, buffer=color_shm3.buf
        )
        depth_img3 = np.ndarray(
            shape=depth_shape, dtype=np.uint16, buffer=depth_shm3.buf
        )

        lock.acquire()
        parts_poses[:] = detection[0]
        parts_found[:] = detection[1]
        color_img1[:] = detection[2]
        depth_img1[:] = detection[3]
        color_img2[:] = detection[4]
        depth_img2[:] = detection[5]
        color_img3[:] = detection[6]
        depth_img3[:] = detection[7]
        lock.release()


@set_frequency(config["furniture"]["detection_hz"])
def _get_parts_poses(
    parts,
    num_parts,
    april_tag,
    cam1,
    cam2,
    cam3,
    cam1_to_base,
    cam2_to_base,
    cam3_to_base,
):
    """
    Args:
        use_base_coord: Use base tag coordinates. If False, use camera coordinate.
    """
    max_fail = 1
    part_idx = 0
    fail_count = 0
    parts_poses = np.zeros(
        (num_parts * 7,), dtype=np.float32
    )  # 3d positional, 4d rotational (quaternion).
    parts_founds = []

    (
        color_img1,
        depth_img1,
        color_img2,
        depth_img2,
        color_img3,
        depth_img3,
        tags1,
        tags2,
        tags3,
    ) = read_detect(april_tag, cam1, cam2, cam3)

    for part in parts:
        part_idx = part.part_idx

        cam1_pose = None
        cam2_pose = _get_parts_pose(part, tags2)
        cam3_pose = _get_parts_pose(part, tags3)
        if cam1_pose is not None:
            cam1_pose = cam1_to_base @ cam1_pose
        if cam2_pose is not None:
            cam2_pose = cam2_to_base @ cam2_pose
        if cam3_pose is not None:
            cam3_pose = cam3_to_base @ cam3_pose

        if cam1_pose is not None or cam2_pose is not None or cam3_pose is not None:
            pose1 = (
                part.pose_filter[0].filter(cam1_pose) if cam1_pose is not None else None
            )
            pose2 = (
                part.pose_filter[1].filter(cam2_pose) if cam2_pose is not None else None
            )
            pose3 = (
                part.pose_filter[2].filter(cam3_pose) if cam3_pose is not None else None
            )
            pose = comp_avg_pose([pose1, pose2, pose3])

            parts_poses[part_idx * 7 : (part_idx + 1) * 7] = np.concatenate(
                list(T.mat2pose(pose))
            ).astype(np.float32)
            part_idx += 1
            parts_founds.append(True)
        else:
            # Detection failed.
            fail_count += 1
            if fail_count >= max_fail:
                part_idx += 1
                parts_founds.append(False)
            else:
                time.sleep(0.01)
                # Read camera and detect tags again.
                (
                    color_img1,
                    depth_img1,
                    color_img2,
                    depth_img2,
                    color_img3,
                    depth_img3,
                    tags1,
                    tags2,
                    tags3,
                ) = read_detect()
        cam1_pose = inv(cam1_to_base) @ cam1_pose if cam1_pose is not None else None
        cam2_pose = inv(cam2_to_base) @ cam2_pose if cam2_pose is not None else None
        cam3_pose = inv(cam3_to_base) @ cam3_pose if cam3_pose is not None else None

    parts_founds = np.array(parts_founds, dtype=bool)
    return (
        parts_poses,
        parts_founds,
        color_img1,
        depth_img1,
        color_img2,
        depth_img2,
        color_img3,
        depth_img3,
    )


def detect_front_rear(parts, num_parts, cam2_to_base, cam3_to_base, tags2, tags3):
    parts_poses = np.zeros(
        (num_parts * 7,), dtype=np.float32
    )  # 3d positional, 4d rotational (quaternion).
    parts_founds = []

    for part in parts:
        part_idx = part.part_idx

        cam2_pose = _get_parts_pose(part, tags2)
        cam3_pose = _get_parts_pose(part, tags3)

        if cam2_to_base is not None and cam2_pose is not None:
            cam2_pose = cam2_to_base @ cam2_pose
        if cam3_to_base is not None and cam3_pose is not None:
            cam3_pose = cam3_to_base @ cam3_pose

        if cam2_pose is not None or cam3_pose is not None:
            pose2 = (
                part.pose_filter[1].filter(cam2_pose) if cam2_pose is not None else None
            )
            pose3 = (
                part.pose_filter[2].filter(cam3_pose) if cam3_pose is not None else None
            )
            if pose2 is not None and pose3 is not None:
                pose = comp_avg_pose([pose2, pose3])
            elif pose2 is not None:
                pose = pose2
            elif pose3 is not None:
                pose = pose3
            else:
                pose = None

            if pose is not None:
                parts_poses[part_idx * 7 : (part_idx + 1) * 7] = np.concatenate(
                    list(T.mat2pose(pose))
                ).astype(np.float32)
                parts_founds.append(True)
            else:
                parts_founds.append(False)
            part_idx += 1

    parts_founds = np.array(parts_founds, dtype=bool)
    return parts_poses, parts_founds


def _get_parts_pose(part, tags):
    poses = []
    for tag_id in part.tag_ids:
        tag = tags.get(tag_id)
        if tag is None:
            continue
        # Get 3d position of 3 points.
        pose = T.to_homogeneous(tag.pose_t, tag.pose_R)
        # Compute where the anchor tag is.
        pose = pose @ np.linalg.inv(part.rel_pose_from_center[tag.tag_id])

        poses.append(pose)
    if len(poses) == 0:
        return None
    return comp_avg_pose(poses)
