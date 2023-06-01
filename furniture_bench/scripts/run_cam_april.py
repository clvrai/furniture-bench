"""Visualize AprilTag detection from three cameras."""
import argparse
import numpy as np
import cv2

from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.perception.apriltag import AprilTag
from furniture_bench.utils.draw import draw_tags
from furniture_bench.config import config


def detect_draw(april_tag, cam: RealsenseCam):
    color_frame, depth_frame = cam.get_frame()
    depth_image = np.asanyarray(depth_frame.get_data()).copy()

    img = np.asanyarray(color_frame.get_data()).copy()

    tags = april_tag.detect(color_frame, cam.intr_param)
    # Visualize tags.
    draw_image = draw_tags(img.copy(), cam, tags)

    return draw_image, depth_image


def main():
    # Define arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-depth", action="store_true", help="Show depth image.")
    args = parser.parse_args()

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
        None,
        disable_auto_exposure=True,
    )
    cam3 = RealsenseCam(
        config["camera"][3]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
    )
    april_tag = AprilTag(tag_size=0.0195)

    cv2.namedWindow("RealsenseAprilTag", cv2.WINDOW_AUTOSIZE)

    while True:
        color_img1, depth_image1 = detect_draw(april_tag, cam1)
        color_img2, depth_image2 = detect_draw(april_tag, cam2)
        color_img3, depth_image3 = detect_draw(april_tag, cam3)
        color_img = np.hstack([color_img1, color_img2, color_img3])
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        if args.show_depth:
            depth_image1 = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image1, alpha=0.1), cv2.COLORMAP_JET
            )
            depth_image2 = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image2, alpha=0.1), cv2.COLORMAP_JET
            )
            depth_image3 = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image3, alpha=0.1), cv2.COLORMAP_JET
            )

            depth_image1 = cv2.resize(
                depth_image1, (color_img1.shape[1], color_img1.shape[0])
            )
            depth_image2 = cv2.resize(
                depth_image2, (color_img1.shape[1], color_img2.shape[0])
            )
            depth_image3 = cv2.resize(
                depth_image3, (color_img1.shape[1], color_img3.shape[0])
            )
            depth_img = np.hstack([depth_image1, depth_image2, depth_image3])
            img = np.vstack([color_img, depth_img])
        else:
            img = color_img
        cv2.imshow("Detected tags", img)

        img2 = cv2.cvtColor(color_img2, cv2.COLOR_RGB2BGR)

        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
