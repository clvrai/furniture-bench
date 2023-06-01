import numpy as np

from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.device.keyboard_interface import KeyboardInterface
from furniture_bench.device.oculus_interface import OculusInterface
import furniture_bench.utils.transform as T


class KeyboardOculusInterface(DeviceInterface):
    """Interface using both Keyboard and Oculus."""

    def __init__(self):
        self.keyboard_interface = KeyboardInterface()
        self.oculus_interface = OculusInterface()

    def get_action(self, use_quat=True):
        keyboard_action, keyboard_done = self.keyboard_interface.get_action()
        oculus_action, oculus_done = self.oculus_interface.get_action()

        pos_action = keyboard_action[:3] + oculus_action[:3]
        rot_action = T.quat_multiply(keyboard_action[3:7], oculus_action[3:7])
        gripper_action = 1 if keyboard_action[7] == 1 or oculus_action[7] == 1 else -1
        action = np.concatenate([pos_action, rot_action, np.array([gripper_action])])
        action = np.clip(action, -1, 1)

        # Use keyboard done.
        if keyboard_done != CollectEnum.DONE_FALSE:
            return action, keyboard_done

        return action, oculus_done

    @property
    def rew_key(self):
        return self.keyboard_interface.rew_key

    def print_usage(self):
        self.keyboard_interface.print_usage()
        self.oculus_interface.print_usage()

    def close(self):
        self.keyboard_interface.close()
        self.oculus_interface.close()
