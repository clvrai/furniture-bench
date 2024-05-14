from furniture_bench.furniture.desk import Desk


class OneLegDesk(Desk):
    def __init__(self):
        super().__init__()
        self.should_be_assembled = [(0, 4)]
