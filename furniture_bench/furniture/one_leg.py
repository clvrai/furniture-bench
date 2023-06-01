from furniture_bench.furniture.square_table import SquareTable


class OneLeg(SquareTable):
    def __init__(self):
        super().__init__()
        self.should_be_assembled = [(0, 4)]
