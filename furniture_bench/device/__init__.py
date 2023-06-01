def make_device(device_name):
    if device_name == "keyboard":
        from furniture_bench.device.keyboard_interface import KeyboardInterface

        device = KeyboardInterface()

    elif device_name == "oculus":
        from furniture_bench.device.oculus_interface import OculusInterface

        device = OculusInterface()

    elif device_name == "keyboard-oculus":
        from furniture_bench.device.keyboard_oculus_interface import (
            KeyboardOculusInterface,
        )

        device = KeyboardOculusInterface()

    else:
        raise Exception(
            f"Unrecognized device: {device}. Choose one of 'keyboard', 'oculus', 'keyboard-oculus'"
        )

    device.print_usage()
    return device
