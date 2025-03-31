import time
from datetime import datetime
from multiprocessing import shared_memory
from typing import List

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


def get_cam_to_base(cam=None, img=None, cam_intr=None, april_tag=None):
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



def detection_loop(config, parts, num_parts, tag_size, lock, shm, num_cameras, no_apriltag):
    print("Start detection")
    
    # Initialize cameras and their transforms dynamically
    cameras = []
    cam_to_bases = []

    for i in range(num_cameras):
        cameras.append(RealsenseCam(
            config["camera"][i + 1]["serial"], # i + 1 because the camera id starts from 1
            config["camera"]["color_img_size"],
            config["camera"]["depth_img_size"],
            config["camera"]["frame_rate"],
            config["camera"][i + 1].get("roi", None),  # ROI is optional
        ))
        # Initialize transforms
        if no_apriltag:
            cam_to_bases.append(None)
        else:
            if i == 0 or i == 1:
                cam_to_bases.append(None)
            else:
                cam_to_bases.append(get_cam_to_base(cameras[i]))
    
    print("Camera initialized")

    april_tag = None if no_apriltag else AprilTag(tag_size)
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
            cameras,
            cam_to_bases,
        )

        # Create shared memory objects dynamically
        shm_objects = []
        for i in range(len(shm)):
            shm_objects.append(shared_memory.SharedMemory(name=shm[i]))

        # Create numpy arrays from shared memory
        parts_poses = np.ndarray(shape=(num_parts * 7,), dtype=np.float32, buffer=shm_objects[0].buf)
        parts_found = np.ndarray(shape=(num_parts,), dtype=bool, buffer=shm_objects[1].buf)
        
        # Create image arrays dynamically
        color_imgs = []
        depth_imgs = []
        for i in range(num_cameras):
            color_imgs.append(np.ndarray(shape=color_shape, dtype=np.uint8, buffer=shm_objects[2+i*2].buf))
            depth_imgs.append(np.ndarray(shape=depth_shape, dtype=np.uint16, buffer=shm_objects[3+i*2].buf))

        # Update shared memory with detection results
        lock.acquire()
        parts_poses[:] = detection[0]
        parts_found[:] = detection[1]
        for i in range(num_cameras):
            color_imgs[i][:] = detection[2+i*2]
            depth_imgs[i][:] = detection[3+i*2]
        lock.release()




@set_frequency(config["furniture"]["detection_hz"])
def _get_parts_poses(
    parts,
    num_parts,
    april_tag,
    cameras: List[RealsenseCam],
    cam_to_bases,
):
    """
    Args:
        use_base_coord: Use base tag coordinates. If False, use camera coordinate.
    """
    parts_poses = np.zeros(
        (num_parts * 7,), dtype=np.float32
    )  # 3d positional, 4d rotational (quaternion).
    parts_founds = []

    color_imgs = []
    depth_imgs = []
    tags = []
    for cam in cameras:
        color_img, depth_img = cam.get_image()
        color_imgs.append(color_img)
        depth_imgs.append(depth_img)
        if april_tag is not None:
            tags.append(april_tag.detect_id(color_img, cam.intr_param))
        else:
            tags.append(None)

    for part in parts:
        part_idx = part.part_idx
        cam_poses = []
        for i, cam in enumerate(cameras):
            cam_pose = _get_parts_pose(part, tags[i])
            if cam_to_bases[i] is not None and cam_pose is not None:
                cam_pose = cam_to_bases[i] @ cam_pose
                cam_pose = part.pose_filter[i].filter(cam_pose)
                cam_poses.append(cam_pose)
        if len(cam_poses) > 0:
            pose = comp_avg_pose(cam_poses)
        else:
            pose = None
        if pose is not None:
            parts_poses[part_idx * 7 : (part_idx + 1) * 7] = np.concatenate(
                list(T.mat2pose(pose))
            ).astype(np.float32)
            parts_founds.append(True)
        else:
            parts_founds.append(False)


    # Interleave the color and depth images
    color_depth_imgs = []
    for i in range(len(color_imgs)):
        color_depth_imgs.append(color_imgs[i])
        color_depth_imgs.append(depth_imgs[i])

    # Return the results
    return (
        parts_poses,
        parts_founds,
        *color_depth_imgs,
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
    if tags is None:
        return None
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
