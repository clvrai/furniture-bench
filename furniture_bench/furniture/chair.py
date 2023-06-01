import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.chair_leg import ChairLeg
from furniture_bench.furniture.parts.chair_nut import ChairNut
from furniture_bench.furniture.parts.chair_seat import ChairSeat
from furniture_bench.furniture.parts.chair_back import ChairBack
from furniture_bench.config import config


class Chair(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["chair"]
        self.furniture_conf = furniture_conf

        self.name = "chair"
        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            ChairSeat(furniture_conf["chair_seat"], 0),
            ChairLeg(furniture_conf["chair_leg1"], 1),
            ChairLeg(furniture_conf["chair_leg2"], 2),
            ChairBack(furniture_conf["chair_back"], 3),
            ChairNut(furniture_conf["chair_nut1"], 4),
            ChairNut(furniture_conf["chair_nut2"], 5),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        self.should_assembled_first[(0, 4)] = (0, 3)
        self.should_assembled_first[(0, 5)] = (0, 3)
        self.assembled_rel_poses[(0, 1)] = [
            get_mat([-0.03375, 0.045, -0.01875], [0, np.pi / 2, 0]),
            get_mat([0.03375, 0.045, -0.01875], [0, np.pi / 2, 0]),
        ]
        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 3)] = [get_mat([0, -0.0325, 0.05025], [0, 0, 0])]
        self.assembled_rel_poses[(0, 4)] = [
            get_mat([0.035, 0, 0.0795], [-np.pi / 2, 0, 0]),
            get_mat([-0.035, 0, 0.0795], [-np.pi / 2, 0, 0]),
        ]
        self.assembled_rel_poses[(0, 5)] = self.assembled_rel_poses[(0, 4)]
