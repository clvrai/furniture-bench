"""Show a trajectory of collected data."""
import argparse
import os
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import cv2

from furniture_bench.utils.detection import get_cam_to_base, detect_front_rear
from furniture_bench.furniture import furniture_factory
from furniture_bench.perception.apriltag import AprilTag
from furniture_bench.config import config
from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.utils.draw import draw_axis
import furniture_bench.utils.transform as T


def main():
    sns.set(rc={"figure.figsize": (8.5 * 3, 4 * 3)})

    # sns.set_style("dark")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to collected data", required=True)
    parser.add_argument("--furniture", required=True)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    furniture = furniture_factory(args.furniture)
    dirs = os.listdir(args.data_dir)

    # cam2 = RealsenseCam(config['camera'][2]['serial'], config['camera']['color_img_size'],
    #                     config['camera']['depth_img_size'], config['camera']['frame_rate'],
    #                     config['camera'][2]['roi'])
    # cam3 = RealsenseCam(config['camera'][3]['serial'], config['camera']['color_img_size'],
    #                     config['camera']['depth_img_size'], config['camera']['frame_rate'])
    cam2_intr = [
        912.234130859375,
        910.6622314453125,
        641.658935546875,
        373.2242736816406,
    ]
    cam3_intr = [
        922.8222045898438,
        923.2533569335938,
        636.3359985351562,
        371.34222412109375,
    ]

    furniture_poses = []
    for dir in dirs[:1]:
        front_path = os.path.join(
            data_dir, dir, os.path.basename(dir) + "_color_image2.mp4"
        )
        rear_path = os.path.join(
            data_dir, dir, os.path.basename(dir) + "_color_image3.mp4"
        )

        front_vid = imageio.get_reader(front_path, "ffmpeg")
        rear_vid = imageio.get_reader(rear_path, "ffmpeg")

        front_image = front_vid.get_data(0)  # Get first frame
        rear_image = rear_vid.get_data(0)

        cam2_to_base = get_cam_to_base(None, 2, front_image, cam2_intr)
        cam3_to_base = get_cam_to_base(None, 3, rear_image, cam3_intr)

        april_tag = AprilTag(0.0195)
        tags2 = april_tag.detect_id(front_image, cam2_intr)
        tags3 = april_tag.detect_id(rear_image, cam3_intr)

        poses, founds = detect_front_rear(
            furniture.parts,
            furniture.num_parts,
            cam2_to_base,
            cam3_to_base,
            tags2,
            tags3,
        )

        for i, found in enumerate(founds):
            if not found:
                continue
            if i == 0:
                name = "chair_seat"
            elif i == 1:
                name = "chair_leg1"
            elif i == 2:
                name = "chair_leg2"
            elif i == 3:
                name = "chair_back"
            elif i == 4:
                name = "chair_nut1"
            elif i == 5:
                name = "chair_nut2"

            furniture_poses.append(
                [name, poses[i * 7 : (i + 1) * 7][0], poses[i * 7 : (i + 1) * 7][1]]
            )
        #     if cam2_to_base is None:
        #         front_image = draw_axis(
        #             front_image,
        #             (np.linalg.inv(cam2_to_base) @ T.pose2mat(poses[i * 7:(i + 1) * 7]))[:3, :3],
        #             (np.linalg.inv(cam2_to_base) @ T.pose2mat(poses[i * 7:(i + 1) * 7]))[:3, 3],
        #             cam2.intr,
        #             s=0.03,
        #             d=4,
        #             rgb=True)
        #     if cam3_to_base is None:
        #         rear_image = draw_axis(
        #             rear_image,
        #             (np.linalg.inv(cam3_to_base) @ T.pose2mat(poses[i * 7:(i + 1) * 7]))[:3, :3],
        #             (np.linalg.inv(cam3_to_base) @ T.pose2mat(poses[i * 7:(i + 1) * 7]))[:3, 3],
        #             cam3.intr,
        #             s=0.03,
        #             d=4,
        #             rgb=True)
        # import pdb; pdb.set_trace()
        # color_img = np.hstack([front_image, rear_image])
        # color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('t', color_img)
        # cv2.waitKey(1)

    # sns.set_palette("hls", 8)
    # sns.set_palette("Set3")

    sns.set(font_scale=4)

    sns.set_style("white", {"axes.linewidth": 1})
    df = pd.DataFrame(furniture_poses, columns=["parts", "x (meter)", "y (meter)"])
    # for i, poses in enumerate(furniture_poses):
    #     for pose in poses:
    ax = sns.scatterplot(
        data=df,
        x="x (meter)",
        y="y (meter)",
        hue="parts",
        palette=sns.color_palette("bright", len(furniture.parts)),
        s=120,
    )

    # ax.get_legend().remove()
    # plt.xlim((-0.22, 0.221))
    plt.xlim((-0.22 + 0.01, 0.221 + 0.01))
    plt.ylim((-0.05, 0.351))

    ax.tick_params(bottom=True, left=True)
    ax.xaxis.set_ticks(list(np.arange(-0.20, 0.201, 0.1)))
    ax.yaxis.set_ticks(list(np.arange(-0.05, 0.351, 0.1)))

    # ax.legend(loc='upper left', handletextpad=0.1, ncol=3)
    plt.legend(
        ncol=6,
        loc="upper center",
        handletextpad=0.1,
        mode="expand",
        fontsize="24",
        markerscale=1.5,
    )
    #  for lh in ax._legend.legendHandles:
    #     lh.set_alpha(1)
    #     lh._sizes = [50]
    # You can also use lh.set_sizes([50])
    # plt.axis('off')

    # ax.set_box_aspect(1)
    # sns.despine(trim=True, offset=2)
    # ax.tight_layout()
    # ax.set(xlabel='x (meter)', ylabel='y (meter)')

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])

    # plt.tight_layout()

    ax.invert_yaxis()

    plt.show()


if __name__ == "__main__":
    main()
