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
    # sns.palplot('palette')
    # sns.set(font_scale=16)
    sns.set_style("darkgrid")
    # sns.set_palette("Set3", 10)

    sns.set(rc={"figure.figsize": (10 * 4, 3 * 4), "font.size": 32})
    # sns.set(rc={'figure.figsize':(10.1,8.27),  "font.size":8})
    # sns.set(rc={"font.size":8})
    sns.set(font_scale=3.5)

    # plt.ylim((0, 1.0))

    df = pd.read_csv("furniture_bench/scripts/plot/reproducibility.csv")
    ax = sns.barplot(
        data=df,
        x="Sites",
        y="Normalized Performance",
        palette=sns.color_palette("Set2"),
    )
    import matplotlib.ticker as mtick

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # plt.legend(ncol=7, loc="upper right", mode='expand')
    # ax.legend()
    # plt.setp(ax.get_legend().get_texts(), fontsize='10')

    # Remove legend title
    # ax.legend_.set_title(None)

    # ax.get_legend().remov()

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    ax.set(xlabel=None)
    plt.show()


if __name__ == "__main__":
    main()
