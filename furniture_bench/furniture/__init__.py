from furniture_bench.furniture.cabinet import Cabinet
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.one_leg import OneLeg
from furniture_bench.furniture.stool import Stool
from furniture_bench.furniture.square_table import SquareTable
from furniture_bench.furniture.round_table import RoundTable
from furniture_bench.furniture.drawer import Drawer
from furniture_bench.furniture.chair import Chair
from furniture_bench.furniture.desk import Desk
from furniture_bench.furniture.lamp import Lamp


def furniture_factory(furniture: str) -> Furniture:
    if furniture == "square_table":
        return SquareTable()
    elif furniture == "desk":
        return Desk()
    elif furniture == "round_table":
        return RoundTable()
    elif furniture == "drawer":
        return Drawer()
    elif furniture == "chair":
        return Chair()
    elif furniture == "lamp":
        return Lamp()
    elif furniture == "cabinet":
        return Cabinet()
    elif furniture == "stool":
        return Stool()
    elif furniture == "one_leg":
        return OneLeg()
    else:
        raise ValueError(f"Unknown furniture type: {furniture}")
