import pyrealsense2 as rs
import numpy as np
import cv2
from dt_apriltags import Detector


class AprilTag:
    def __init__(self, tag_size):
        self.at_detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            # quad_sigma=0.0,
            # refine_edges=1,
            # decode_sharpening=0.25,
            debug=0,
        )
        self.tag_size = tag_size

    def detect(self, frame, intr_param):
        """Detect AprilTag.

        Args:
            frame: pyrealsense2.frame or Gray-scale image to detect AprilTag.
            intr_param: Camera intrinsics format of [fx, fy, ppx, ppy].
        Returns:
            Detected tags.
        """
        if isinstance(frame, rs.frame):
            frame = np.asanyarray(frame.get_data())
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        detections = self.at_detector.detect(frame, True, intr_param, self.tag_size)
        # Filter out bad detections.
        return [detection for detection in detections if detection.hamming < 2]

    def detect_id(self, frame, intr_param):
        detections = self.detect(frame, intr_param)
        # Make it as a dictionary which the keys are tag_id.
        return {detection.tag_id: detection for detection in detections}
