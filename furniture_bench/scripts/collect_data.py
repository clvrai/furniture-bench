import os
import argparse
import os.path as osp

import furniture_bench
from furniture_bench.device import make_device
from furniture_bench.data.data_collector import DataCollector
from furniture_bench.config import config
from furniture_bench.envs.initialization_mode import Randomness


def main():
    parser = argparse.ArgumentParser(description="Collect IL data")
    parser.add_argument(
        "--out-data-path", help="Path to directory to save the data", required=True
    )
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
        default="keyboard-oculus",
    )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        choices=list(config["furniture"].keys()),
        required=True,
    )
    parser.add_argument(
        "--is-sim",
        action="store_true",
        help="Use simulator, else use real world environment.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use scripted function for getting action.",
    )
    parser.add_argument(
        "--env",
        type=str,
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        help="Use learned policy to collect the data.",
    )
    parser.add_argument(
        "--ckpt-step",
        type=int,
    )
    parser.add_argument(
        "--pkl-only",
        action="store_true",
        help="Only save the pickle file, not .mp4 and .pngs",
    )
    parser.add_argument(
        "--save-failure",
        action="store_true",
        help="Save failure trajectories.",
    )
    parser.add_argument(
        "--headless", help="With front camera view", action="store_true"
    )
    parser.add_argument(
        "--draw-marker", action="store_true", help="Draw AprilTag marker"
    )
    parser.add_argument(
        "--manual-label",
        action="store_true",
        help="Manually label the reward",
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )

    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
    )

    parser.add_argument("--num-demos", default=100, type=int)

    parser.add_argument("--resize-sim-img", action="store_true")
    parser.add_argument("--gripper-pos-control", action="store_true")


    args = parser.parse_args()

    if args.scripted:
        assert args.is_sim
        device_interface = None
    else:
        device_interface = make_device(args.input_device)

    data_path = osp.join(args.out_data_path, args.furniture)
    if not osp.isdir(data_path):
        os.makedirs(data_path)

    data_collector = DataCollector(
        env=args.env,
        is_sim=args.is_sim,
        data_path=data_path,
        device_interface=device_interface,
        furniture=args.furniture,
        headless=args.headless,
        draw_marker=args.draw_marker,
        manual_label=args.manual_label,
        scripted=args.scripted,
        randomness=args.randomness,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        pkl_only=args.pkl_only,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        resize_sim_img=args.resize_sim_img,
        gripper_pos_control=args.gripper_pos_control,
        ckpt_dir=args.ckpt_dir,
        ckpt_step=args.ckpt_step
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
