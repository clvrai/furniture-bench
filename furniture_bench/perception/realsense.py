import time

import numpy as np
import pyrealsense2 as rs

from furniture_bench.perception.apriltag import AprilTag


class RealsenseCam:
    def __init__(
        self,
        serial,
        color_res,
        depth_res,
        frame_rate,
        roi=None,
        disable_auto_exposure: bool = False,
    ):
        self.started = False
        self.serial = serial

        if serial is None:
            from furniture_bench.config import config

            # Find which camera's serial is not set.
            for i in range(1, 4):
                if config["camera"][i]["serial"] is None:
                    raise ValueError(
                        f" Camera {i} serial is not set. \n Run export CAM{i}_SERIAL=<serial> before running this script. \n "
                    )

        self.color_res = color_res
        self.depth_res = depth_res
        self.frame_rate = frame_rate
        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()
        # Configure streams
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, *self.color_res, rs.format.rgb8, self.frame_rate
        )
        config.enable_stream(
            rs.stream.depth, *self.depth_res, rs.format.z16, self.frame_rate
        )
        # Start streaming
        self.roi = roi
        try:
            conf = self.pipeline.start(config)
        except Exception as e:
            print(f"[Error] Could not initialize camera serial: {self.serial}")
            raise e

        self.min_depth = 0.15
        self.max_depth = 2.0
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, self.min_depth)
        self.threshold_filter.set_option(rs.option.max_distance, self.max_depth)

        # Get intrinsic parameters of color image``.
        profile = conf.get_stream(
            rs.stream.color
        )  # Fetch stream profile for depth stream
        intr_param = (
            profile.as_video_stream_profile().get_intrinsics()
        )  # Downcast to video_stream_profile and fetch intrinsics
        self.intr_param = [intr_param.fx, intr_param.fy, intr_param.ppx, intr_param.ppy]
        self.intr_mat = self._get_intrinsic_matrix()

        # Get the sensor once at the beginning. (Sensor index: 1)
        color_sensor = conf.get_device().first_color_sensor()
        # Set the exposure anytime during the operation
        color_sensor.set_option(rs.option.enable_auto_exposure, True)

        # Set region of interest.
        # color_sensor = conf.get_device().first_roi_sensor()

        if disable_auto_exposure:
            color_sensor.set_option(rs.option.enable_auto_exposure, False)

        if roi is not None:
            # Disable auto exposure.

            # TODO: Fix this.
            roi_sensor = color_sensor.as_roi_sensor()
            roi = roi_sensor.get_region_of_interest()
            roi.min_x, roi.max_x = self.roi[0], self.roi[1]
            roi.min_y, roi.max_y = self.roi[2], self.roi[3]
            # https://github.com/IntelRealSense/librealsense/issues/8004
            roi_success = False
            for _ in range(5):
                try:
                    roi_sensor.set_region_of_interest(roi)
                except:
                    time.sleep(0.1)
                    pass
                else:
                    roi_success = True
                    break
            if not roi_success:
                print("Could not set camera ROI.")

        for _ in range(10):
            # Read dummy observation to setup exposure.
            self.get_frame()
            time.sleep(0.04)

        self.started = True

    def get_frame(self):
        """Read frame from the realsense camera.

        Returns:
            Tuple of color and depth image. Return None if failed to read frame.

            color frame:(height, width, 3) RGB uint8 realsense2.video_frame.
            depth frame:(height, width) z16 realsense2.depth_frame.
        """
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_frame = self.threshold_filter.process(depth_frame)

        if not color_frame or not depth_frame:
            return None, None
        return color_frame, depth_frame

    def get_image(self):
        """Get numpy color and depth image.

        Returns:
            Tuble of numpy color and depth image. Return (None, None) if failed.

            color image: (height, width, 3) RGB uint8 numpy array.
            depth image: (height, width) z16 numpy array.
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None or depth_frame is None:
            return None, None
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        return color_image, depth_image

    def _get_intrinsic_matrix(self):
        m = np.zeros((3, 3))
        m[0, 0] = self.intr_param[0]
        m[1, 1] = self.intr_param[1]
        m[0, 2] = self.intr_param[2]
        m[1, 2] = self.intr_param[3]
        return m

    def __del__(self):
        if self.started:
            self.pipeline.stop()


def frame_to_image(color_frame, depth_frame):
    color_image = np.asanyarray(color_frame.get_data()).copy()
    depth_image = np.asanyarray(depth_frame.get_data()).copy()

    return color_image, depth_image


def read_detect(
    april_tag: AprilTag, cam1: RealsenseCam, cam2: RealsenseCam, cam3: RealsenseCam
):
    color_img1, depth_img1 = cam1.get_image()
    color_img2, depth_img2 = cam2.get_image()
    color_img3, depth_img3 = cam3.get_image()
    tags1 = april_tag.detect_id(color_img1, cam1.intr_param)
    tags2 = april_tag.detect_id(color_img2, cam2.intr_param)
    tags3 = april_tag.detect_id(color_img3, cam3.intr_param)

    return (
        color_img1,
        depth_img1,
        color_img2,
        depth_img2,
        color_img3,
        depth_img3,
        tags1,
        tags2,
        tags3,
    )
