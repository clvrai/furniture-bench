from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl

fig = plt.figure()
ax = plt.axes(projection="3d")
# fig = plt.figure()
# ax = fig.gca(projection='3d')

fixed_paths = [
    "/hdd/IL_data/one_leg/2022-12-09-14:09:13/2022-12-09-14:09:13.pkl",
    "/hdd/IL_data/one_leg/2022-12-09-14:10:04/2022-12-09-14:10:04.pkl",
    "/hdd/IL_data/one_leg/2022-12-09-14:38:53/2022-12-09-14:38:53.pkl",
]

range_random_paths = [
    "/hdd/IL_data_rand/one_leg/2022-12-16-17:40:25/2022-12-16-17:40:25.pkl",
    "/hdd/IL_data_rand/one_leg/2022-12-16-17:39:02/2022-12-16-17:39:02.pkl",
    "/hdd/IL_data_rand/one_leg/2022-12-16-16:56:39/2022-12-16-16:56:39.pkl",
]

paths = range_random_paths


max_len = -np.inf
for path in paths:
    with open(path, "rb") as f:
        data = pickle.load(f)

    ee_x = []
    ee_y = []
    ee_z = []
    for obs in data["observations"]:
        ee_x.append(obs["robot_state"]["ee_pos"][0])
        ee_y.append(obs["robot_state"]["ee_pos"][1])
        ee_z.append(obs["robot_state"]["ee_pos"][2])

    N = len(ee_x)

    step = 1
    for i in range(0, N - 1, step):
        ax.plot(
            ee_x[i : i + step + 1],
            ee_y[i : i + step + 1],
            ee_z[i : i + step + 1],
            color=plt.cm.jet(int(255 * i / N)),
        )

    if len(ee_x) > max_len:
        max_len = len(ee_x)

norm = mpl.colors.Normalize(vmin=0, vmax=max_len)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))

fig.colorbar(cmap, ticks=np.arange(max_len, step=100), shrink=0.8)

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()
