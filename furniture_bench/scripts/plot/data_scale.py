import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    # sns.set_palette("Paired”)
    # sns.set_palette("hsl", 8)
    # sns.set_palette("Set2")
    # Save a palette to a variable:
    # sns.set_palette("bright")
    # sns.set_palette("Paired")
    # Use palplot and pass in the variable:
    # sns.palplot(palette)
    # sns.set(font_scale=16)
    # sns.set_palette("Set3", 10)

    sns.set(rc={"figure.figsize": (11.7 * 2, 8.27 * 2), "font.size": 32})
    # sns.set(rc={'figure.figsize':(10.1,8.27),  "font.size":8})
    # sns.set(rc={"font.size":8})
    sns.set(font_scale=2)

    sns.set_style("darkgrid")
    d = pd.read_csv("data_scale.csv")
    ax = sns.lineplot(
        x="Demos",
        y="Success Rate",
        linestyle="-",
        # markers=True,
        markers=["o", "o", "o", "o"],
        hue="Method-Env",
        style="Method-Env",
        linewidth=3,
        sizes=[10, 10, 10, 10],
        data=d,
    )
    # palette="pastel")
    plt.setp(ax.get_legend().get_texts(), fontsize="25")
    plt.setp(ax.get_legend().get_title(), fontsize="25")
    ax.set_xlabel("Num Demos", fontsize=40)
    ax.set_ylabel("Success Rate", fontsize=40)

    plt.show()


if __name__ == "__main__":
    main()
