import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os

markers = ["o", "^", ".", "*", "X"]
line_styles = ["--", "solid", "dotted", "dashdot"]
patterns = ["", "\\", "", "\\"]
x = [100, 250, 500, 750, 1000]

sns.set_style("whitegrid")

# colors = ['#5EA3EF', '#5EA3EF', '#EF8636', '#B671DE', '#458933', '#FF5968', '#FBE5A3']
# colors = ["#5EA3EF", "#EF8636", "#B671DE", "#458933", "#FF5968", "#FBE5A3"]
colors = ["#5EA3EF", "#E41A1C", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B", "#F28E2B"]


# phase_colors = ['#E3E3E3', '#BABDC1', '#A2A6AC', '#8C939A', '#5D6874']
phase_colors = ["#E3E3E3", "#BABDC1", "#E3E3E3", "#BABDC1", "#E3E3E3"]
# phase_colors = ['#C5E8B7', '#ABE098', '#83D475', '#57C84D', '#2EB62C']

# custom_palette = sns.color_palette(colors)
# sns.color_palette(colors)
custom_palette= sns.color_palette(sns.color_palette("husl", 11) + ["#B671DE"])
# custom_palette = sns.color_palette(["#5EA3EF"] + sns.color_palette("RdPu", 11))

custom_palette = [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.8836443049112893, 0.5240073524369634, 0.19569304285113343), (0.710130687316902, 0.6046852192663268, 0.19426060163712158), (0.5432776721247529, 0.6540981095185215, 0.19324494273892204), (0.19592059105779686, 0.6981620017487838, 0.3452219818913641), (0.2067117296964458, 0.6829103404254792, 0.5829988925822328), (0.21420912437215422, 0.6714963557258681, 0.6986206664203177), (0.22537170008202412, 0.6531400148480775, 0.841007805313343), (0.5596943802099308, 0.5764402169887779, 0.9583930713150347), (0.8578978803740231, 0.44058452715322166, 0.957819659566579), "#B671DE"]


matplotlib.rcParams["hatch.linewidth"] = 4

df = pd.read_csv(f"{os.path.dirname(__file__)}/reproducibility.csv")

# from seaborn.categorical import _BarPlotter
# _BarPlotter.width = 0.8  # Change the value to adjust the bar width

plt.figure(figsize=(12, 8))

sns.barplot(
    x="Setup",
    y="Score",
    data=df,
    errorbar=("pi", 100),
    # ci=100,
    capsize=0.4,
    errwidth=4,
    width=0.8,
    palette=custom_palette,
)
plt.ylabel("Completed phases", fontsize=25, labelpad=10)
plt.xlabel("")
plt.xticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["Original", "", "", "", "", "", "Reproduced [1-10]", "", "", "", ""], fontsize=25
)
plt.ylim(0.8, 5.2)
plt.yticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], fontsize=26)
plt.tight_layout(w_pad=1.5)
# plt.show()

plt.savefig("reproducibility.pdf")
