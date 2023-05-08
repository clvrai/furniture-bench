"""Define the enum for data collection"""
from enum import Enum


class CollectEnum(Enum):
    """Define the enum for data collection"""

    DONE_FALSE = 2  # Data collect is not done yet
    SUCCESS = 3  # Data collect is done and success
    FAIL = 4  # Data collect is done and fail
    REWARD = 5  # Reward key is pressed
    SKILL = 6  # Skill key is pressed
    RESET = 7  # Reset key is pressed
