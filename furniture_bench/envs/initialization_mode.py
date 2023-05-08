"""Define randomize mode for furniture environment."""
from enum import Enum


class Randomness(Enum):
    LOW_RANDOM = 0
    MEDIUM_RANDOM = 1
    HIGH_RANDOM = 2
    MEDIUM_COLLECT = 3
    HIGH_RANDOM_COLLECT = 4
    SKILL_FIXED = 5
    SKILL_RANDOM = 6


def str_to_enum(str):
    if str == "low":
        # Low initialization randomness
        return Randomness.LOW_RANDOM
    elif str == "med":
        # Medium initialization randomness
        return Randomness.MEDIUM_RANDOM
    elif str == "high":
        # High initialization randomness
        return Randomness.HIGH_RANDOM
    elif str == "med_collect":
        # Medium initialization randomness for data collection
        # (i.e., no strict pre-defined initial pose, but check range)
        return Randomness.MEDIUM_COLLECT
    elif str == "skill_fixed":
        return Randomness.SKILL_FIXED
    elif str == "skill_random":
        return Randomness.SKILL_RANDOM
    else:
        raise ValueError("Unknown initialization mode: {}".format(str))
