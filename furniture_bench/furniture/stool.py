import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.config import config
from furniture_bench.furniture.parts.stool_leg import StoolLeg
from furniture_bench.furniture.parts.stool_seat import StoolSeat


class Stool(Furniture):
    def __init__(self):
        super().__init__()
        self.name = "stool"
        furniture_conf = config["furniture"]["stool"]
        self.furniture_conf = furniture_conf

        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            StoolSeat(furniture_conf["stool_seat"], 0),
            StoolLeg(furniture_conf["stool_leg1"], 1),
            StoolLeg(furniture_conf["stool_leg2"], 2),
            StoolLeg(furniture_conf["stool_leg3"], 3),
        ]
        self.num_parts = len(self.parts)

        self.assembled_rel_poses = {}

        # TODO: Need accurate rel_pose.
        self.assembled_rel_poses[(0, 1)] = [
            get_mat([0, 0.045, 0.040542], [0, np.pi / 2, 0]),
            get_mat([0, 0, 0], [0, 2 * np.pi / 3, 0])
            @ get_mat([0, 0.045, 0.040542], [0, np.pi / 2, 0]),
            get_mat([0, 0, 0], [0, 4 * np.pi / 3, 0])
            @ get_mat([0, 0.045, 0.040542], [0, np.pi / 2, 0]),
        ]
        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 3)] = self.assembled_rel_poses[(0, 1)]

        self.should_be_assembled = [(0, 1), (0, 2), (0, 3)]
        self.skill_attach_part_idx = 3

    def get_grasp_part_idx(self, from_skill):
        if from_skill == 1:
            return 0
        elif from_skill == 3:
            return 3
        else:
            assert False
