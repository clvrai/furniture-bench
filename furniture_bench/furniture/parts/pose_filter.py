from collections import deque

from furniture_bench.utils.pose import is_similar_pose


class PoseFilter:
    """Conduct filtering to compensate noise of Apriltag detection."""

    def __init__(self):
        self.pose_queue = deque(maxlen=5)

    def filter(self, pose):
        if len(self.pose_queue) == 0:
            self.pose_queue.append(pose)
            return pose

        recent_diff_pose = None
        off_counter = 0
        for p in reversed(self.pose_queue):
            if not is_similar_pose(
                p, pose, ori_bound=0.6, pos_threshold=[0.05, 0.05, 0.05]
            ):
                off_counter += 1
                # Store most recent different pose.
                if recent_diff_pose is None:
                    recent_diff_pose = p

        # Safe detecion.
        if off_counter <= (len(self.pose_queue) // 2):
            res_pose = pose
        else:
            # Abnormal pose. Use pose in the queue.
            res_pose = recent_diff_pose

        # There could be two source of "off" pose detecion.
        # 1. Due to noise of camera and Apriltag detection.
        # 2. Due to actual dramatic change of pose (e.g. robot arm hit the object).
        # Thus, save the new pose anyway, to handle those situations above.
        self.pose_queue.append(pose)

        return res_pose

    def reset(self):
        self.pose_queue.clear()
