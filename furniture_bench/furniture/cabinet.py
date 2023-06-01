import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.cabinet_body import CabinetBody
from furniture_bench.furniture.parts.cabinet_door_left import CabinetDoorLeft
from furniture_bench.furniture.parts.cabinet_door_right import CabinetDoorRight
from furniture_bench.furniture.parts.cabinet_top import CabinetTop
from furniture_bench.config import config


class Cabinet(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["cabinet"]
        self.furniture_conf = furniture_conf

        self.name = "cabinet"
        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            CabinetBody(furniture_conf["cabinet_body"], 0),
            CabinetDoorLeft(furniture_conf["cabinet_door_left"], 1),
            CabinetDoorRight(furniture_conf["cabinet_door_right"], 2),
            CabinetTop(furniture_conf["cabinet_top"], 3),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 2), (0, 1), (0, 3)]

        self.should_assembled_first[(0, 3)] = (0, 2)
        self.should_assembled_first[(0, 3)] = (0, 1)

        self.assembled_rel_poses[(0, 2)] = [
            get_mat([-0.0275, -0.0375, -0.025], [0, np.pi / 2, 0])
        ]
        self.assembled_rel_poses[(0, 1)] = [
            get_mat([-0.02275, -0.0375, 0.025], [0, np.pi / 2, 0])
        ]
        self.assembled_rel_poses[(0, 3)] = [
            get_mat([0.0, -0.07750, 0], [0, -np.pi / 2, 0])
        ]
