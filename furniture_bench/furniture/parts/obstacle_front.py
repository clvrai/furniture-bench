from furniture_bench.utils.pose import get_mat, rot_mat


class ObstacleFront:
    def __init__(self):
        self.reset_pos = [[0.0, 0.35 + 0.01, 0]]
        self.reset_ori = [rot_mat([0, 0, 0], hom=True)]
        self.reset_x_len = 0.35
        self.reset_y_len = 0.02
        self.mut_ori = rot_mat([0, 0, 0], hom=True)
