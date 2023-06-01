from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.desk_leg import DeskLeg
from furniture_bench.furniture.parts.desk_table_top import DeskTableTop
from furniture_bench.config import config


class Desk(Furniture):
    def __init__(self):
        super().__init__()
        self.name = "desk"
        furniture_conf = config["furniture"]["desk"]
        self.furniture_conf = furniture_conf
        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            DeskTableTop(furniture_conf["desk_top"], 0),
            DeskLeg(furniture_conf["leg1"], 1),
            DeskLeg(furniture_conf["leg2"], 2),
            DeskLeg(furniture_conf["leg3"], 3),
            DeskLeg(furniture_conf["leg4"], 4),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 1), (0, 2), (0, 3), (0, 4)]

        self.assembled_rel_poses[(0, 1)] = [
            get_mat([-0.080, 0.065625, -0.05], [0, 0, 0]),
            get_mat([-0.080, 0.065625, 0.05], [0, 0, 0]),
            get_mat([0.080, 0.065625, -0.05], [0, 0, 0]),
            get_mat([0.080, 0.065625, 0.05], [0, 0, 0]),
        ]

        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 3)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 4)] = self.assembled_rel_poses[(0, 1)]
