import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    sns.set(rc={"figure.figsize": (10 * 4, 3 * 4), "font.size": 32})
    # sns.set(rc={'figure.figsize':(10.1,8.27),  "font.size":8})
    # sns.set(rc={"font.size":8})
    sns.set(font_scale=2)
    sns.set_style("darkgrid")

    plt.ylim((0, 0.5))

    df = pd.read_csv("bar1.csv")
    ax = sns.barplot(data=df, x="Furniture", y="Performance", hue="Metric")

    plt.legend(ncol=2, loc="upper right", columnspacing=0.5)
    # plt.legend(ncol=3, loc="upper right", mode='expand')
    plt.setp(ax.get_legend().get_texts(), fontsize="12")

    # Remove legend title
    ax.legend_.set_title(None)

    # ax.get_legend().remov()

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])

    ax.set(xlabel=None)
    plt.show()


if __name__ == "__main__":
    main()
