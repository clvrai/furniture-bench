import keyboard

from furniture_bench.device.keyboard_interface import KeyboardInterface
from furniture_bench.device.keyboard_oculus_interface import KeyboardOculusInterface
from furniture_bench.device.oculus_interface import OculusInterface


def make_device_interface(device_name):
    if device_name == "keyboard":
        keyboard_interface = KeyboardInterface()
        keyboard.on_press(keyboard_interface.on_press)
        device = keyboard_interface
    elif device_name == "oculus":
        device = OculusInterface()
    elif device_name == "keyboard-oculus":
        device = KeyboardOculusInterface()
    else:
        raise Exception(f"Unrecognized device: {device}")

    device.print_usage()
    return device
