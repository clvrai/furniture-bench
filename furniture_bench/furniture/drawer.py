import numpy as np

from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.drawer_box import DrawerBox
from furniture_bench.furniture.parts.drawer_container import DrawerContainer
from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat


class Drawer(Furniture):
    def __init__(self):
        super().__init__()
        furniture_conf = config["furniture"]["drawer"]
        self.furniture_conf = furniture_conf

        self.name = "drawer"
        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            DrawerBox(furniture_conf["drawer_box"], 0),
            DrawerContainer(furniture_conf["drawer_container_top"], 1),
            DrawerContainer(furniture_conf["drawer_container_bottom"], 2),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 2), (0, 1)]

        self.assembled_rel_poses[(0, 1)] = [
            get_mat([0, -0.0345, 0.008], [0, 0, 0]),
            get_mat([0, 0.0105, 0.008], [0, 0, 0]),
        ]
        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
