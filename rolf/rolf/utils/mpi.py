import numpy as np
from mpi4py import MPI

from .info_dict import Info


def mpi_gather_average(x):
    buf = MPI.COMM_WORLD.gather(x, root=0)
    if MPI.COMM_WORLD.rank == 0:
        info = Info()
        for data in buf:
            info.add(data)
        return info.get_dict()
    return None


def _mpi_average(x):
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    buf /= MPI.COMM_WORLD.Get_size()
    return buf


# Average across the cpu's data
def mpi_average(x):
    if MPI.COMM_WORLD.Get_size() == 1:
        return x
    if isinstance(x, dict):
        keys = sorted(x.keys())
        return {k: _mpi_average(np.array(x[k])) for k in keys}
    else:
        return _mpi_average(np.array(x))


def _mpi_sum(x):
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    return buf


# Sum over the cpu's data
def mpi_sum(x):
    if MPI.COMM_WORLD.Get_size() == 1:
        return x
    if isinstance(x, dict):
        keys = sorted(x.keys())
        return {k: _mpi_sum(np.array(x[k])) for k in keys}
    else:
        return _mpi_sum(np.array(x))


# Syncronize all processes.
def mpi_sync():
    mpi_sum(0)
