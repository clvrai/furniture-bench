from furniture_bench.utils.pose import get_mat
from furniture_bench.config import config
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.square_table_leg import SquareTableLeg
from furniture_bench.furniture.parts.square_table_top import SquareTableTop


class SquareTable(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["square_table"]
        self.furniture_conf = furniture_conf

        self.tag_size = furniture_conf["tag_size"]

        self.parts = [
            SquareTableTop(furniture_conf["square_table_top"], 0),
            SquareTableLeg(furniture_conf["square_table_leg1"], 1),
            SquareTableLeg(furniture_conf["square_table_leg2"], 2),
            SquareTableLeg(furniture_conf["square_table_leg3"], 3),
            SquareTableLeg(furniture_conf["square_table_leg4"], 4),
        ]
        self.num_parts = len(self.parts)

        self.assembled_rel_poses[(0, 1)] = [
            get_mat([-0.05625, 0.046875, -0.05625], [0, 0, 0]),
            get_mat([0.05625, 0.046875, -0.05625], [0, 0, 0]),
            get_mat([-0.05625, 0.046875, 0.05625], [0, 0, 0]),
            get_mat([0.05625, 0.046875, 0.05625], [0, 0, 0]),
        ]

        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 3)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 4)] = self.assembled_rel_poses[(0, 1)]

        self.should_be_assembled = [(0, 4), (0, 3), (0, 1), (0, 2)]
        self.skill_attach_part_idx = 4

    def get_grasp_part_idx(self, from_skill):
        if from_skill == 1:
            return 0
        elif from_skill == 3:
            return 4
        else:
            assert False

    def z_noise(self, from_skill):
        # Zero noise for collision.
        return 0
