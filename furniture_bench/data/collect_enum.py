"""Define the enum for data collection."""

from enum import Enum


class CollectEnum(Enum):
    DONE_FALSE = 2  # Data collection in progress.
    SUCCESS = 3  # Successful trajectory is collected.
    FAIL = 4  # Failed trajectory is collected.
    REWARD = 5  # Annotate reward +1.
    SKILL = 6  # Annotate new skill.
    RESET = 7  # Reset environment.
    TERMINATE = 8  # Terminate data collection.
