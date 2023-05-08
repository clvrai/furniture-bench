import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

furn_dict = {
    "cabinet": 11,
    "chair": 17,
    "drawer": 8,
    "desk": 16,
    "square_table": 16,
    "stool": 11,
    "round_table": 8,
    "lamp": 7,
    "one_leg": 5,
}

phase_dict = {
    "cabinet": [4, 6, 11],
    "chair": [5, 8, 11, 14, 17],
    "drawer": [5, 8],
    "desk": [5, 8, 13, 16],
    "square_table": [5, 8, 13, 16],
    "stool": [5, 8, 11],
    "round_table": [5, 8],
    "lamp": [5, 7],
    "one_leg": [5],
}

markers = ["o", "^", ".", "*", "X"]
line_styles = ["--", "solid", "dotted", "dashdot"]
patterns = ["", "\\", "", "\\"]
x = [100, 250, 500, 750, 1000]

sns.set_style("whitegrid")

colors = ["#5EA3EF", "#EF8636", "#B671DE", "#458933", "#FF5968", "#FBE5A3"]
# phase_colors = ['#E3E3E3', '#BABDC1', '#A2A6AC', '#8C939A', '#5D6874']
# phase_colors = ['#EEEEEE', '#82D580', '#EEEEEE', '#82D580', '#EEEEEE']
phase_colors = ["#EEEEEE", "#C3C3C3", "#EEEEEE", "#C3C3C3", "#EEEEEE"]
custom_palette = sns.color_palette(colors)
# phase_colors = ['#E3E3E3', '#BABDC1', '#E3E3E3', '#BABDC1', '#E3E3E3']
# phase_colors = ['#C5E8B7', '#ABE098', '#83D475', '#57C84D', '#2EB62C']
sns.color_palette(phase_colors)

matplotlib.rcParams["hatch.linewidth"] = 4

for furniture in furn_dict.keys():
    FURNITURE = furniture
    SUBTASKS = furn_dict[FURNITURE]
    phases = phase_dict[FURNITURE]

    import os

    df = pd.read_csv(f"{os.path.dirname(__file__)}/data/{FURNITURE}.csv")
    df.dropna(axis=0, inplace=True)
    print(FURNITURE)
    sns.set_theme(style="whitegrid")
    sns.set_palette(custom_palette)

    fig, ax = plt.subplots(figsize=(5, 5))

    margin = 0.1
    for i, phase in enumerate(phases):
        if i == 0:
            ax.axhspan(0, phase, facecolor=phase_colors[i], alpha=0.9)
        else:
            ax.axhspan(phases[i - 1], phase, facecolor=phase_colors[i], alpha=0.9)

    # ax.axhspan(phases[i-1] + margin, phase, facecolor=phase_colors[i], alpha=1.0)

    bar = sns.barplot(
        x="Randomness",
        y="Score",
        data=df,
        hue="Algorithm",
        errorbar=("pi", 100),
        capsize=0.2,
        errwidth=4,
    )

    # for i,thisbar in enumerate(bar.patches):
    #     thisbar.set_hatch(patterns[i])

    plt.title(f"{FURNITURE}", fontsize=26, pad=10)
    plt.xlabel("Randomness", fontsize=24)
    plt.ylabel("Completed phases", fontsize=24)

    margin = 0.05 if FURNITURE == "one_leg" else 0
    plt.ylim(bottom=-0.01, top=SUBTASKS + margin)
    plt.xticks([0, 1, 2], ["Low", "Medium", "High"], fontsize=24)
    if FURNITURE in ["one_leg", "round_table", "drawer", "lamp"]:
        plt.yticks([i for i in range(0, SUBTASKS + 1, 2)] + [SUBTASKS], fontsize=24)
    elif FURNITURE in ["chair", "cabinet", "stool"]:
        plt.yticks([i for i in range(0, SUBTASKS + 1, 3)] + [SUBTASKS], fontsize=24)
    else:
        plt.yticks([i for i in range(0, SUBTASKS + 1, 4)] + [SUBTASKS], fontsize=24)

    # plt.legend(loc='center left', bbox_to_anchor=(0.3,-0.2))
    bar.get_legend().remove()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{os.path.dirname(__file__)}/plots/full/{FURNITURE}.pdf")
