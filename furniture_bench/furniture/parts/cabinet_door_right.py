import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.parts.cabinet_door import CabinetDoor


class CabinetDoorRight(CabinetDoor):
    def __init__(self, part_config, part_idx):
        super().__init__(part_config, part_idx)

        self.handle_rel_pose = get_mat([-0.01, 0.035, 0], [0, 0, 0])
