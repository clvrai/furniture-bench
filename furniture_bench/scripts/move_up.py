"""Move the robot to initial pose."""
from furniture_bench.robot.panda import Panda
from furniture_bench.config import config


def main():
    robot = Panda(config["robot"])
    print("Reset the robot")
    robot.reset()
    robot.z_move(0.2)
    print("Reset done.")


if __name__ == "__main__":
    main()
