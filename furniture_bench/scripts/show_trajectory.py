"""Show a trajectory of collected data."""
import time
import glob
import argparse
import pickle
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2


def get_frames_from_video(video_path):
    video = cv2.VideoCapture(str(video_path))
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to collected data")
    parser.add_argument(
        "--data-path", help="Path to collected data of single pickle file."
    )
    parser.add_argument(
        "--channel-first", help="Path to collected data .pkl", action="store_true"
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--show-raw-images", action="store_true", help="Show original images."
    )
    parser.add_argument(
        "--speed-up", type=int, default=5, help="Speed up the video by this factor."
    )

    np.set_printoptions(suppress=True)

    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        file = os.path.join(args.data_dir, data_dir.name + ".pkl")
    elif args.data_path is not None:
        file = args.data_path
    else:
        raise ValueError("Either data_dir or data_path must be specified.")

    if args.save_video:
        path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".avi"
        size = (224 * 2, 224)
        out = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, size, True
        )

    with open(file, "rb") as f:
        data = pickle.load(f)
        len_traj = len(data["actions"])
        rewards = []
        sum_skills = 0

        if args.data_dir is not None:
            data_dir = Path(args.data_dir)

        if args.data_dir is not None:
            if args.show_raw_images:
                # Read raw RGB images and depth images.
                depth_images1 = []
                depth_images2 = []
                depth_images3 = []

                s = time.time()
                print("Start reading depth images...")
                depth_image1 = sorted(
                    glob.glob(str(data_dir / f"{data_dir.name}_depth_image1" / "*.png"))
                )
                depth_image2 = sorted(
                    glob.glob(str(data_dir / f"{data_dir.name}_depth_image2" / "*.png"))
                )
                depth_image3 = sorted(
                    glob.glob(str(data_dir / f"{data_dir.name}_depth_image3" / "*.png"))
                )
                for i in range(len_traj):
                    depth_images1.append(cv2.imread(depth_image1[i], -1))
                    depth_images2.append(cv2.imread(depth_image2[i], -1))
                    depth_images3.append(cv2.imread(depth_image3[i], -1))
                print(time.time() - s)

                s = time.time()
                print(
                    "Start reaAssertionError: The observation returned by the `reset()` method is not contained with the observation spaceding color images..."
                )
                s = time.time()

                video_paths = [
                    data_dir / f"{data_dir.name}_color_image1.mp4",
                    data_dir / f"{data_dir.name}_color_image2.mp4",
                    data_dir / f"{data_dir.name}_color_image3.mp4",
                ]
                color_images1, color_images2, color_images3 = [
                    get_frames_from_video(video_path) for video_path in video_paths
                ]

        for i in range(len_traj):
            if "color_image1" in data["observations"][0]:
                color_image1 = data["observations"][i]["color_image1"]
                color_image2 = data["observations"][i]["color_image2"]
                if args.show_raw_images:
                    org_color_image1 = color_images1[i]
                    org_color_image2 = color_images2[i]
                    org_color_image3 = color_images3[i]

                    org_depth_image1 = depth_images1[i]
                    org_depth_image2 = depth_images2[i]
                    org_depth_image3 = depth_images3[i]
                    org_depth_image1 = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_images1[i], alpha=0.1),
                        cv2.COLORMAP_JET,
                    )
                    org_depth_image2 = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_images2[i], alpha=0.1),
                        cv2.COLORMAP_JET,
                    )
                    org_depth_image3 = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_images3[i], alpha=0.1),
                        cv2.COLORMAP_JET,
                    )
            else:
                # It is converted data.
                color_image1 = data["observations"][i]["image1"]
                color_image2 = data["observations"][i]["image2"]

            if args.channel_first:
                color_image1 = np.moveaxis(color_image1, 0, -1)
                color_image2 = np.moveaxis(color_image2, 0, -1)

            color_image1 = cv2.cvtColor(color_image1, cv2.COLOR_RGB2BGR)
            color_image2 = cv2.cvtColor(color_image2, cv2.COLOR_RGB2BGR)
            color_img = np.hstack([color_image1, color_image2])
            if data["rewards"][i] != 0:
                rewards.append(data["rewards"][i])
            print(data["observations"][i]["robot_state"])
            cv2.putText(
                np.ascontiguousarray(color_img),
                "rewards: " + str(rewards),
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 0),
            )

            if data["skills"][i] != 0:
                sum_skills += data["skills"][i]

            cv2.putText(
                np.ascontiguousarray(color_img),
                "skills: " + str(sum_skills),
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 0, 255),
            )

            cv2.imshow("Trajectory", color_img)
            if args.show_raw_images:
                # imageio read RGB.
                org_color_image1 = cv2.cvtColor(org_color_image1, cv2.COLOR_RGB2BGR)
                org_color_image2 = cv2.cvtColor(org_color_image2, cv2.COLOR_RGB2BGR)
                org_color_image3 = cv2.cvtColor(org_color_image3, cv2.COLOR_RGB2BGR)
                org_color_img = np.hstack(
                    [org_color_image1, org_color_image2, org_color_image3]
                )
                org_depth_img = np.hstack(
                    [org_depth_image1, org_depth_image2, org_depth_image3]
                )
                org_img = np.vstack([org_color_img, org_depth_img])
                cv2.imshow(
                    "Original Trajectory", cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
                )

            if args.save_video:
                out.write(color_img)

            sleep_time = 0.1 / args.speed_up
            time.sleep(sleep_time)
            k = cv2.waitKey(1)
            if k == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
