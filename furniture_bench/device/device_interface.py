from typing import Tuple
from abc import ABC, abstractmethod

import numpy.typing as npt

from furniture_bench.data.collect_enum import CollectEnum


class DeviceInterface(ABC):
    @abstractmethod
    def get_action(self, use_quat: bool = True) -> Tuple[npt.NDArray, CollectEnum]:
        """Return action from the io device and whether the episode is done."""
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
