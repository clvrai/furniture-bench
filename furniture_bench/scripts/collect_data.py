import os
import argparse
import os.path as osp

from furniture_bench.data.data_collector import DataCollector
from furniture_bench.device import make_device_interface
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
        "--scripted",
        action="store_true",
        help="Use scripted function for getting action.",
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
        "--headless", help="With front camera view", action="store_true"
    )
    parser.add_argument("--draw-marker", action="store_true")
    parser.add_argument(
        "--manual-label", help="Manually label the reward", action="store_true"
    )
    parser.add_argument(
        "--random-init", default="low", choices=["low", "medium", "high"]
    )
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-demos", default=100, type=int)
    args = parser.parse_args()

    if args.scripted:
        assert args.is_sim
        device_interface = None
    else:
        device_interface = make_device_interface(args.input_device)

    data_path = osp.join(args.out_data_path, args.furniture)
    if not osp.isdir(data_path):
        os.makedirs(data_path)

    if args.random_init == "low":
        randomness = Randomness.LOW_RANDOM
    elif args.random_init == "medium":
        randomness = Randomness.MEDIUM_COLLECT
    else:
        randomness = Randomness.HIGH_RANDOM_COLLECT
    data_collector = DataCollector(
        args.is_sim,
        data_path,
        device_interface,
        args.furniture,
        args.headless,
        args.draw_marker,
        args.manual_label,
        args.scripted,
        randomness,
        args.gpu_id,
        args.num_demos,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
