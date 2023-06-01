import time
import logging

import numpy as np
import colorlog


formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "white",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

Logger = colorlog.getLogger("robot-learning")
Logger.setLevel(logging.DEBUG)
Logger.propagate = False

# fh = logging.FileHandler('log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# Logger.addHandler(fh)

if not Logger.handlers:
    ch = colorlog.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    Logger.addHandler(ch)


class StopWatch(object):
    def __init__(self):
        self.start = {}
        self.times = {}

    def begin(self, name):
        self.start[name] = time.time()

    def end(self, name):
        if name not in self.times:
            self.times[name] = []
        assert name in self.start, "%s cannot be found in Stop Watch" % name

        self.times[name].append(time.time() - self.start[name])

    def display(self):
        print("----Times----")
        for name in self.times:
            print(name, np.mean(self.times[name]))

        self.times = {}
