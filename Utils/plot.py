import matplotlib.pyplot as plt


def kde_plot(df, save_path, xlabel, ylabel, xlimit=None,
             figsize=(6.5, 5)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    df.plot.kde(ax=ax, legend=True, xlim=xlimit)
    ax.legend(fontsize=14)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.savefig(save_path, bbox_inches="tight") 