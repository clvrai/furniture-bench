import argparse
import gym
import cv2
import numpy as np

import furniture_bench


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help="Number of cameras to use (0-4)",
    )
    parser.add_argument(
        "--no_apriltag", action="store_true", help="Disable AprilTag detection"
    )
    parser.add_argument(
        "--no_visualize", action="store_true", help="Disable visualization"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(
        "FurnitureBench-v0",
        furniture="cabinet",
        resize_img=True,
        manual_done=False,
        with_display=True,
        draw_marker=False,
        manual_label=False,
        randomness="high",
        high_random_idx=-1,
        visualize_init_pose=True,
        record=False,
        manual_reset=False,
        num_cameras=args.num_cameras,
        no_apriltag=args.no_apriltag,
    )

    print("Resetting environment...")
    env.reset()

    # Zero actions.
    for _ in range(1000):
        ac = np.array([0, 0, 0, 0, 0, 0, 1, -1])
        ob, rew, done, _ = env.step(ac)

        # Collect and stack images horizontally
        if args.num_cameras > 0 and not args.no_visualize:
            images = []
            for i in range(1, args.num_cameras + 1):
                camera_img = ob[f"color_image{i}"][..., ::-1]  # Convert BGR to RGB
                images.append(camera_img)

            # Horizontally stack all images
            stacked_image = np.hstack(images)

            # Display the stacked images
            cv2.imshow("All Cameras", stacked_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
