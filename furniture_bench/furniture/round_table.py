import numpy as np

from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.round_table_top import RoundTableTop
from furniture_bench.furniture.parts.round_table_leg import RoundTableLeg
from furniture_bench.furniture.parts.round_table_base import RoundTableBase
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.config import config
import furniture_bench.utils.transform as T


class RoundTable(Furniture):
    def __init__(self):
        super().__init__()

        self.name = "round_table"
        furniture_conf = config["furniture"]["round_table"]
        self.furniture_conf = furniture_conf
        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            RoundTableTop(furniture_conf["round_table_top"], 0),
            RoundTableLeg(furniture_conf["round_table_leg"], 1),
            RoundTableBase(furniture_conf["round_table_base"], 2),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 1), (1, 2)]

        self.assembled_rel_poses[(0, 1)] = [
            # get_mat([0, 0, 0.044375], [np.pi / 2, 0, np.pi + np.pi / 36])
            get_mat([0, 0, 0.044375], [np.pi / 2, 0, np.pi + np.pi / 36])
        ]
        self.assembled_rel_poses[(1, 2)] = [
            get_mat([0, 0.053125, 0], [-np.pi / 2, np.pi / 2, 0])
        ]

        self.skill_attach_part_idx = 1

    def get_grasp_part_idx(self, from_skill):
        if from_skill == 1:
            return 0
        elif from_skill == 3:
            return 1
        else:
            assert False

    def z_noise(self, from_skill):
        # Zero noise for collision.
        if from_skill == 4:
            return 0
        return None
