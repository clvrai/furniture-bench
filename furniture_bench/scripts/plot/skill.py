import os
import argparse
import seaborn as sns
from pathlib import Path

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to collected data", required=True)
    parser.add_argument("--skill", type=int, required=True)
    args = parser.parse_args()

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    files = list(Path(args.data_dir).rglob("*.pkl"))
    if len(files) == 0:
        raise Exception("Data path is empty")

    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    z_min, z_max = np.inf, -np.inf
    for i, file in enumerate(sorted(files)):
        # if i == 10:
        #     break

        print(f"[{i + 1} / {len(files)}] reading {file}")
        with open(file, "rb") as f:
            data = pickle.load(f)
        # ee_x = []
        # ee_y = []
        # ee_z = []
        sum_skill = 0

        for i, obs in enumerate(data["observations"]):
            if data["skills"][i] == 1:
                sum_skill += 1
            # ee_x.append(obs['robot_state']['ee_pos'][0])
            # ee_y.append(obs['robot_state']['ee_pos'][1])
            # ee_z.append(obs['robot_state']['ee_pos'][2])
            if sum_skill == args.skill:
                # cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
                ax.scatter(
                    obs["robot_state"]["ee_pos"][0],
                    obs["robot_state"]["ee_pos"][1],
                    obs["robot_state"]["ee_pos"][2],
                    c="blue",
                    marker="o",
                )
                x_max = max(x_max, obs["robot_state"]["ee_pos"][0])
                x_min = min(x_min, obs["robot_state"]["ee_pos"][0])
                y_max = max(y_max, obs["robot_state"]["ee_pos"][1])
                y_min = min(y_min, obs["robot_state"]["ee_pos"][1])
                z_max = max(z_max, obs["robot_state"]["ee_pos"][2])
                z_min = min(z_min, obs["robot_state"]["ee_pos"][2])
                break

        # print(f"X min max: {}, {}".format(x_min, x_max))
    print("X min max:  {0:0.3f}, {1:0.3f}".format(x_min, x_max))
    print("Y min max:  {0:0.3f}, {1:0.3f}".format(y_min, y_max))
    print("Z min max:  {0:0.3f}, {1:0.3f}".format(z_min, z_max))
    # print(f"Y min max: {}, {}".format(x_min, x_max))
    # print(f"Z min max: {}, {}".format(x_min, x_max))
    # N = len(ee_x)

    # step = 1
    # for i in range(0, N - 1, step):
    # ax.plot(ee_x[i:i + step + 1],
    #         ee_y[i:i + step + 1],
    #         ee_z[i:i + step + 1],
    #         color=plt.cm.jet(int(255 * i / N)))

    # if len(ee_x) > max_len:
    #     max_len = len(ee_x)

    # norm = mpl.colors.Normalize(vmin=0, vmax=max_len)
    # cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    # cmap.set_array([])

    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # fig.colorbar(cmap, ticks=np.arange(max_len, step=100),shrink=0.8)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()


if __name__ == "__main__":
    main()
