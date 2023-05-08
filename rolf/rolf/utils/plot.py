from collections import namedtuple, defaultdict
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import wandb


matplotlib.rcParams["pdf.fonttype"] = 42  # Important!!! Remove Type 3 fonts


def save_fig(file_name, file_format="pdf", tight=True, **kwargs):
    if tight:
        plt.tight_layout()
    file_name = "{}.{}".format(file_name, file_format).replace(" ", "-")
    plt.savefig(file_name, format=file_format, dpi=1000, **kwargs)


def draw_line(
    log,
    method,
    avg_step=3,
    mean_std=False,
    max_step=None,
    max_y=None,
    x_scale=1.0,
    ax=None,
    color="C0",
    smooth_steps=10,
    num_points=50,
    line_style="-",
    marker=None,
    no_fill=False,
    smoothing_weight=0.0,
):
    steps = {}
    values = {}
    max_step = max_step * x_scale
    seeds = log.keys()
    is_line = True

    for seed in seeds:
        step = np.array(log[seed].steps)
        value = np.array(log[seed].values)

        if not np.isscalar(log[seed].values):
            is_line = False

            # filter NaNs
            for i in range(len(value)):
                if np.isnan(value[i]):
                    value[i] = 0 if i == 0 else value[i - 1]

        if max_step:
            max_step = min(max_step, step[-1])
        else:
            max_step = step[-1]

        steps[seed] = step
        values[seed] = value

    if is_line:
        y_data = [values[seed] for seed in seeds]
        std_y = np.std(y_data)
        avg_y = np.mean(y_data)
        min_y = np.min(y_data)
        max_y = np.max(y_data)

        l = ax.axhline(
            y=avg_y, label=method, color=color, linestyle=line_style, marker=marker
        )
        ax.axhspan(
            avg_y - std_y,  # max(avg_y - std_y, min_y),
            avg_y + std_y,  # min(avg_y + std_y, max_y),
            color=color,
            alpha=0.1,
        )
        return l, min_y, max_y

    # exponential moving average smoothing
    for seed in seeds:
        last = values[seed][:10].mean()  # First value in the plot (first timestep)
        smoothed = list()
        for point in values[seed]:
            smoothed_val = (
                last * smoothing_weight + (1 - smoothing_weight) * point
            )  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value
        values[seed] = smoothed

    # cap all sequences to max number of steps
    data = []
    for seed in seeds:
        for i in range(len(steps[seed])):
            if steps[seed][i] <= max_step:
                data.append((steps[seed][i], values[seed][i]))
    data.sort()
    x_data = []
    y_data = []
    for step, value in data:
        x_data.append(step)
        y_data.append(value)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    min_y = np.min(y_data)
    max_y = np.max(y_data)
    # l = sns.lineplot(x=x_data, y=y_data)
    # return l, min_y, max_y

    # filling
    if not no_fill:
        n = len(x_data)
        avg_step = int(n // num_points)

        x_data = x_data[: n // avg_step * avg_step].reshape(-1, avg_step)
        y_data = y_data[: n // avg_step * avg_step].reshape(-1, avg_step)

        std_y = np.std(y_data, axis=1)

        avg_x, avg_y = np.mean(x_data, axis=1), np.mean(y_data, axis=1)
    else:
        avg_x, avg_y = x_data, y_data

    # subsampling smoothing
    n = len(avg_x)
    ns = smooth_steps
    avg_x = avg_x[: n // ns * ns].reshape(-1, ns).mean(axis=1)
    avg_y = avg_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)
    if not no_fill:
        std_y = std_y[: n // ns * ns].reshape(-1, ns).mean(axis=1)

    if not no_fill:
        ax.fill_between(
            avg_x,
            avg_y - std_y,  # np.clip(avg_y - std_y, 0, max_y),
            avg_y + std_y,  # np.clip(avg_y + std_y, 0, max_y),
            alpha=0.1,
            color=color,
        )

    # horizontal line
    if "SAC" in method:
        l = ax.axhline(
            y=avg_y[-1], xmin=0.1, xmax=1.0, color=color, linestyle="--", marker=marker
        )
        plt.setp(l, linewidth=2, color=color, linestyle="--", marker=marker)

    l = ax.plot(avg_x, avg_y, label=method)
    plt.setp(l, linewidth=2, color=color, linestyle=line_style, marker=marker)
    # 4 if 'Ours' not in method else 2

    return l, min_y, max_y


def draw_graph(
    plot_logs,
    line_logs,
    method_names=None,
    title=None,
    xlabel="Step",
    ylabel="Success",
    legend=False,
    mean_std=False,
    min_step=0,
    max_step=None,
    min_y=None,
    max_y=None,
    num_y_tick=5,
    smooth_steps=10,
    num_points=50,
    no_fill=False,
    num_x_tick=5,
    legend_loc=2,
    markers=None,
    smoothing_weight=0.0,
    file_name=None,
    line_styles=None,
    line_colors=None,
):
    if legend:
        fig, ax = plt.subplots(figsize=(15, 5))
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
    max_value = -np.inf
    min_value = np.inf

    if method_names is None:
        method_names = list(plot_logs.keys()) + list(line_logs.keys())

    lines = []
    num_colors = len(method_names)
    two_lines_per_method = False
    if "Pick" in method_names[0] or "Attach" in method_names[0]:
        two_lines_per_method = True
        num_colors = len(method_names) / 2

    for idx, method_name in enumerate(method_names):
        if method_name in plot_logs.keys():
            log = plot_logs[method_name]
        else:
            log = line_logs[method_name]

        seeds = log.keys()
        if len(seeds) == 0:
            continue

        color = (
            line_colors[method_name] if line_colors else "C%d" % (num_colors - idx - 1)
        )
        line_style = line_styles[method_name] if line_styles else "-"

        l_, min_, max_ = draw_line(
            log,
            method_name,
            mean_std=mean_std,
            max_step=max_step,
            max_y=max_y,
            x_scale=1.0,
            ax=ax,
            color=color,
            smooth_steps=smooth_steps,
            num_points=num_points,
            line_style=line_style,
            no_fill=no_fill,
            smoothing_weight=smoothing_weight[idx]
            if isinstance(smoothing_weight, list)
            else smoothing_weight,
            marker=markers[idx] if isinstance(markers, list) else markers,
        )
        # lines += l_
        max_value = max(max_value, max_)
        min_value = min(min_value, min_)

    if min_y == None:
        min_y = int(min_value - 1)
    if max_y == None:
        max_y = max_value
        # max_y = int(max_value + 1)

    # y-axis tick (belows are commonly used settings)
    if max_y == 1:
        plt.yticks(np.arange(min_y, max_y + 0.1, 0.2), fontsize=12)
    else:
        if max_y > 1:
            plt.yticks(
                np.arange(min_y, max_y + 0.01, (max_y - min_y) / num_y_tick),
                fontsize=12,
            )  # make this 4 for kitchen
        elif max_y > 0.8:
            plt.yticks(np.arange(0, 1.0, 0.2), fontsize=12)
        elif max_y > 0.5:
            plt.yticks(np.arange(0, 0.8, 0.2), fontsize=12)
        elif max_y > 0.3:
            plt.yticks(np.arange(0, 0.5, 0.1), fontsize=12)
        elif max_y > 0.2:
            plt.yticks(np.arange(0, 0.4, 0.1), fontsize=12)
        else:
            plt.yticks(np.arange(0, 0.2, 0.05), fontsize=12)

    # x-axis tick
    plt.xticks(
        np.round(
            np.arange(min_step, max_step + 0.1, (max_step - min_step) / num_x_tick), 2
        ),
        fontsize=12,
    )

    # background grid
    ax.grid(b=True, which="major", color="lightgray", linestyle="--")

    # axis titles
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # set axis range
    ax.set_xlim(min_step, max_step)
    ax.set_ylim(bottom=-0.01, top=max_y + 0.01)  # use -0.01 to print ytick 0

    # print legend
    if legend:
        if isinstance(legend_loc, tuple):
            print("print legend outside of frame")
            leg = plt.legend(fontsize=15, bbox_to_anchor=legend_loc, ncol=6)
        else:
            leg = plt.legend(fontsize=15, loc=legend_loc)

    #         for line in leg.get_lines():
    #             line.set_linewidth(2)
    # labs = [l.get_label() for l in lines]
    # plt.legend(lines, labs, fontsize='small', loc=2)

    # print title
    if title:
        plt.title(title, y=1.00, fontsize=16)

    # save plot to file
    if file_name:
        save_fig(file_name)


def build_logs(
    methods_label,
    runs,
    data_key="train_ep/episode_success",
    x_scale=1000000,
    op=None,
    exclude_runs=[],
):
    Log = namedtuple("Log", ["values", "steps"])
    logs = defaultdict(dict)
    for run_name in methods_label.keys():
        for i, seed_path in enumerate(methods_label[run_name]):
            if seed_path in exclude_runs:
                print("Exclude run: {}".format(seed_path))
                continue

            found_path = False
            for run in runs:
                if run.name == seed_path:
                    data = run.history(samples=10000)
                    values = data[data_key]
                    steps = data["_step"] / x_scale
                    if op == "max":
                        values = max(values)
                    logs[run_name][i] = Log(values, steps)
                    print(run_name, i, run, len(steps))
                    found_path = True
            if not found_path:
                # raise ValueError("Could not find run: {}".format(seed_path))
                print("Could not find run: {}".format(seed_path))
    return logs


def build_logs_pick_attach(
    methods_label, runs, x_scale=1000000, data_key=None, op=None, exclude_runs=[]
):
    Log = namedtuple("Log", ["values", "steps"])
    logs = defaultdict(dict)
    for method_name, method_runs in methods_label.items():
        for i, seed_path in enumerate(method_runs):
            if seed_path in exclude_runs:
                print("Exclude run: {}".format(seed_path))
                continue

            found_path = False
            for run in runs:
                if run.name == seed_path:
                    data = run.history(samples=10000)
                    pick_values = (data[data_key + "phase"] >= 4) + (
                        data[data_key + "phase"] >= 12
                    )
                    attach_values = data[data_key + "success_reward"] / 100
                    steps = data["_step"] / x_scale
                    if op == "max":
                        pick_values = max(pick_values)
                        attach_values = max(attach_values)
                    logs[method_name + "-Pick"][i] = Log(pick_values, steps)
                    logs[method_name + "-Attach"][i] = Log(attach_values, steps)
                    print(method_name, i, run, len(steps))
                    found_path = True

            if not found_path:
                # raise ValueError("Could not find run: {}".format(seed_path))
                print("Could not find run: {}".format(seed_path))
    return logs


def plot_furniture_pick_attach():
    print("** start plot")
    print()

    furnitures = [
        "three_blocks_peg",
        "toy_table",
        "table_bjorkudden_0207",
        "table_dockstra_0279",
        "bench_bjursta_0210",
        "chair_agne_0007",
        "chair_ingolf_0650",
        "table_lack_0825",
    ]
    # furnitures = ["three_blocks_peg"]
    # furnitures = ["chair_ingolf_0650"]
    seeds = [123, 456, 789]
    # seeds = [123]

    exclude_runs = [
        "three_blocks_peg.ppo.0324_test.456",
        "toy_table.ppo.0324_test.456",
        "table_lack_0825.ppo.0324_test.456",
        "chair_ingolf_0650.ppo.0324_test.456",
        "chair_agne_0007.ppo.0324_test.456",
        "bench_bjursta_0210.ppo.0324_test.456",
    ]

    print("** load runs from wandb")
    api = wandb.Api()
    runs = api.runs(
        path="clvr/bimanual"
    )  # , filters={'config.data.dataset_spec.env_name': 'kitchen-mixed-v0'})

    for furniture in furnitures:
        print()
        print("** plot " + furniture)

        plot_labels = {
            "PPO": ["{}.ppo.0324_test.{}".format(furniture, seed) for seed in seeds]
            + ["{}.ppo.0324_test2.{}".format(furniture, seed) for seed in seeds],
            "SAC": ["{}.sac.0324_test.{}".format(furniture, seed) for seed in seeds],
            "GAIL": ["{}.gail.0324_test.{}".format(furniture, seed) for seed in seeds],
            "GAIL+PPO": [
                "{}.gail.0324_test_gail_ppo.{}".format(furniture, seed)
                for seed in seeds
            ],
        }

        line_labels = {
            "BC": ["{}.bc.0324_test.{}".format(furniture, seed) for seed in [123]],
        }

        line_colors = {
            "PPO-Pick": "C4",
            "PPO-Attach": "C4",
            "SAC-Pick": "C3",
            "SAC-Attach": "C3",
            "GAIL-Pick": "C2",
            "GAIL-Attach": "C2",
            "GAIL+PPO-Pick": "C1",
            "GAIL+PPO-Attach": "C1",
            "BC-Pick": "C0",
            "BC-Attach": "C0",
        }
        line_styles = {
            "PPO-Pick": "--",
            "PPO-Attach": "-",
            "SAC-Pick": "--",
            "SAC-Attach": "-",
            "GAIL-Pick": "--",
            "GAIL-Attach": "-",
            "GAIL+PPO-Pick": "--",
            "GAIL+PPO-Attach": "-",
            "BC-Pick": "--",
            "BC-Attach": "-",
        }

        print("** load data from wandb")
        plot_logs = build_logs_pick_attach(
            plot_labels, runs, data_key="train_ep/", exclude_runs=exclude_runs
        )
        line_logs = build_logs_pick_attach(
            line_labels, runs, data_key="test_ep/", op="max", exclude_runs=exclude_runs
        )

        print("** draw graph")
        draw_graph(
            plot_logs,  # curved lines
            line_logs,  # straight line
            method_names=None,  # method names to plot with order
            title=None,  # figure title on top
            xlabel="Environment steps (1M)",  # x-axis title
            ylabel="Average Count",  # y-axis title
            legend=False,
            legend_loc=(1.1, 1.3),  # (1.03, 0.73),
            max_step=20,  # 50,
            min_y=0,
            max_y=1.5,
            num_y_tick=3,
            smooth_steps=1,
            num_points=100,
            num_x_tick=4,  # 5,
            smoothing_weight=0.99,
            file_name=furniture + "_pick_attach_20M",
            line_colors=line_colors,
            line_styles=line_styles,
        )


def plot_furniture():
    print("** start plot")
    print()

    furnitures = [
        "three_blocks_peg",
        "toy_table",
        "table_bjorkudden_0207",
        "table_dockstra_0279",
        "bench_bjursta_0210",
        "chair_agne_0007",
        "chair_ingolf_0650",
        "table_lack_0825",
    ]
    # furnitures = ["table_dockstra_0279"]
    # furnitures = ["three_blocks_peg"]
    # furnitures = ["chair_ingolf_0650"]
    seeds = [123, 456, 789]
    # seeds = [123]

    exclude_runs = [
        "three_blocks_peg.ppo.0324_test.456",
        "toy_table.ppo.0324_test.456",
        "table_lack_0825.ppo.0324_test.456",
        "chair_ingolf_0650.ppo.0324_test.456",
        "chair_agne_0007.ppo.0324_test.456",
        "bench_bjursta_0210.ppo.0324_test.456",
    ]

    ghost_script = True

    print("** load runs from wandb")
    api = wandb.Api()
    runs = api.runs(path="clvr/bimanual")

    for furniture in furnitures:
        print()
        print("** plot " + furniture)

        plot_labels = {
            "PPO": ["{}.ppo.0324_test.{}".format(furniture, seed) for seed in seeds]
            + ["{}.ppo.0324_test2.{}".format(furniture, seed) for seed in seeds],
            "SAC": ["{}.sac.0324_test.{}".format(furniture, seed) for seed in seeds],
            "GAIL": ["{}.gail.0324_test.{}".format(furniture, seed) for seed in seeds],
            "GAIL+PPO": [
                "{}.gail.0324_test_gail_ppo.{}".format(furniture, seed)
                for seed in seeds
            ],
        }

        line_labels = {
            "BC": ["{}.bc.0324_test.{}".format(furniture, seed) for seed in [123]],
        }

        print("** load data from wandb")
        plot_logs = build_logs(
            plot_labels, runs, data_key="train_ep/phase", exclude_runs=exclude_runs
        )
        line_logs = build_logs(
            line_labels,
            runs,
            data_key="test_ep/phase",
            op="max",
            exclude_runs=exclude_runs,
        )

        print("** draw graph")
        draw_graph(
            plot_logs,  # curved lines
            line_logs,  # straight line
            method_names=None,  # method names to plot with order
            title=None,  # figure title on top
            xlabel="Environment steps (1M)",  # x-axis title
            ylabel="Phase",  # y-axis title
            legend=False,  # True if furniture == "three_blocks_peg" else False,
            legend_loc=(1.1, 1.2),  # (1.03, 0.73),
            max_step=20,
            min_y=0,
            max_y=10,
            num_y_tick=5,
            smooth_steps=1,
            num_points=100,
            num_x_tick=4,  # 5,
            smoothing_weight=0.99,
            file_name=furniture + "_20M",
        )

        def gs_opt(filename):
            filename_reduced = filename.split(".")[-2] + "_reduced.pdf"
            gs = [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/default",  # Image resolution
                "-dNOPAUSE",  # No pause after each image
                "-dQUIET",  # Suppress output
                "-dBATCH",  # Automatically exit
                "-dDetectDuplicateImages=true",  # Embeds images used multiple times only once
                "-dCompressFonts=true",  # Compress fonts in the output (default)
                "-r150",
                # '-dEmbedAllFonts=false',
                # '-dSubsetFonts=true',           # Create font subsets (default)
                "-sOutputFile=" + filename_reduced,  # Save to temporary output
                filename,  # Input file
            ]

            subprocess.run(gs)  # Create temporary file
            # subprocess.run(['del', filename],shell=True)            # Delete input file
            # subprocess.run(['ren', filenameTmp,filename],shell=True) # Rename temporary to input file

        if ghost_script:
            gs_opt(furniture + "_20M.pdf")


if __name__ == "__main__":
    plot_furniture()
    # plot_furniture_pick_attach()
