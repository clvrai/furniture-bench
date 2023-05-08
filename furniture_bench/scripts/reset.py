"""Move the robot to initial pose."""
from furniture_bench.robot import Panda
from furniture_bench.config import config


def main():
    robot = Panda(config["robot"])
    print("Reset the robot")
    robot.reset()
    print("Reset done.")


if __name__ == "__main__":
    main()
