from furniture_bench.utils.pose import get_mat, rot_mat


class ObstacleRight:
    def __init__(self):
        self.reset_pos = [[0.175, 0.37 + 0.01 - 0.075, 0]]
        self.reset_ori = [rot_mat([0, 0, 0], hom=True)]
        self.reset_x_len = 0.02
        self.reset_y_len = 0.17

        self.mut_ori = rot_mat([0, 0, 0], hom=True)
