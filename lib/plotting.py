import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

line_styles = 10 * ["-", "--", ":", "-."]
markers = 10 * ["o", "v", "^", "s", "P", "X", "D", "+", "x"]

dash_styles = 10 * ["",
                    (4, 1.5),
                    (1, 1),
                    (3, 1, 1.5, 1),
                    (5, 1, 1, 1),
                    (5, 1, 2, 1, 2, 1),
                    (2, 2, 3, 1.5),
                    (1, 2.5, 3, 1.2)]

PLOT_WIDTH = 5
PLOT_HEIGHT = 3

colors = sns.color_palette()


def set_log_scale(ax, metric, log_rules, force_linear_x=False):
    m = metric.split(':')[-1]
    rule = log_rules.get(m, 'linear')
    if rule == 'linear':
        pass
    elif rule == 'loglog':
        if not force_linear_x:
            ax.set_xscale('log')
        ax.set_yscale('log')
    elif rule == 'logx':
        if not force_linear_x:
            ax.set_xscale('log')
    elif rule == 'logy':
        ax.set_yscale('log')
    else:
        raise ValueError(f"Unknown log rule `{rule}`")


def update_labels(axes, metric_dict):
    def _parse(label):
        return label.split(':')[-1]

    for ax in axes.reshape(-1):
        xlabel = _parse(ax.get_xlabel())
        ylabel = _parse(ax.get_ylabel())

        if xlabel in metric_dict.keys():
            ax.set_xlabel(metric_dict[xlabel])

        if ylabel in metric_dict.keys():
            ax.set_ylabel(metric_dict[ylabel])


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'p_%s' % n
    return percentile_


def plot_logs(logs, path, metrics, main_key, style_key=None, ylims=dict(), log_rules=dict(), metric_dict=dict(),
              **kwargs):
    """
    Make a grid of line plots with std intervals. Each subplot correspond to one metric.
    :param path: output file
    :param logs: dataframe containing all the training data per time step
    :param metrics: metrics to plot (elbo, nll, ...)
    :param main_key: key to used for coloring
    :param style_key: key used for styling (Optional)
    :param ylims: dictionary for the y axis boundaries (Optional)
    :return: None
    """

    N = len(metrics)
    ncols = 2
    nrows = N // ncols

    if N > ncols * nrows:
        nrows += 1
        metrics = list(metrics) + [metrics[-1] for _ in
                                   range((ncols * nrows - len(metrics)))]  # repeat the last plot for the legend

    hue_order = list(logs[main_key].unique())
    step_min = np.percentile(logs['step'].values.tolist(), 10) if len(logs['step']) else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows))
    for i, k in tqdm(list(enumerate(metrics)), desc="|    subplots"):
        u = i // ncols
        v = i % ncols
        ax = axes[u, v]

        sns.lineplot(x="step", y="_value",
                     hue=main_key,
                     hue_order=hue_order,
                     style=style_key,
                     data=logs[logs['_key'] == k],
                     ax=ax,
                     dashes=dash_styles
                     # palette=sns.color_palette("mako_r", 4)
                     )

        # set log scale
        set_log_scale(ax, k, log_rules, force_linear_x=True)

        ax.set_ylabel(k)
        # y lims
        if k in ylims:
            ax.set_ylim(ylims[k])
        elif ax.get_yaxis().get_scale() != 'log':
            ys = logs[(logs['_key'] == k) & (logs['step'] > step_min)]['_value'].values.tolist()
            if len(ys):
                a, b = np.percentile(ys, [25, 75])
                M = b - a
                k = 1.5
                ax.set_ylim([a - k * M, b + k * M])

        if i < len(metrics) - 1:
            ax.get_legend().remove()

    # draw legend in the last plot
    # patches = [mpatches.Patch(color=sns.color_palette()[i], label=key) for i, key in enumerate(hue_order)]
    # axes[nrows - 1, ncols - 1].legend(handles=patches)

    update_labels(axes, metric_dict)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def spot_on_plot(logs, path, metrics, main_key, auxiliary_key, style_key=None, ylims=dict(), log_rules=dict(),
                 metric_dict=dict(), **kwargs):
    """make a grid of plot metrics x aux. keys values"""
    aux_keys = logs[auxiliary_key].unique()
    nrows = len(metrics)
    ncols = len(aux_keys)

    if nrows == 0 or ncols == 0:
        return None

    hue_order = list(logs[main_key].unique())
    step_min = np.percentile(logs['step'].values.tolist(), 10) if len(logs['step']) else 0  # used to filter first steps
    fig, axes = plt.subplots(nrows, ncols, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows), sharex='col',
                             sharey='row')

    for j, aux_key in tqdm(list(enumerate(sorted(aux_keys))), desc="|  aux. keys"):
        aux_data = logs[logs[auxiliary_key] == aux_key]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j] if ncols > 1 else axes[i]
                data = aux_data[aux_data["_key"] == metric]

                sns.lineplot(x="step", y="_value",
                             hue=main_key,
                             hue_order=hue_order,
                             style=style_key,
                             data=data,
                             ax=ax,
                             # palette=sns.color_palette("mako_r", 4)
                             )

                # set log scale
                set_log_scale(ax, metric, log_rules, force_linear_x=True)

                # define axis labels and hide x,y axis in the middle plots
                if i == 0:
                    ax.set_title(f"{auxiliary_key} = {aux_key}")

                if i < len(metrics) - 1:
                    # ax.set_xticklabels([])
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_ylabel(metric)
                else:
                    ax.set_ylabel("")
                    # ax.set_yticklabels([])

                if not (i == len(metrics) - 1 and j == len(aux_keys) - 1):
                    ax.get_legend().remove()

            except:
                warnings.warn(f">> spot-on plot: couldn't generate the axis `ax[{i}, {j}]`")

    # scale y axes
    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        data = logs[logs["_key"] == metric]
        if metric in ylims:
            ax.set_ylim(ylims[metric])
        elif ax.get_yaxis().get_scale() != 'log':
            ys = data[data['step'] > step_min]['_value'].values.tolist()
            if len(ys):
                a, b = np.percentile(ys, [25, 75])
                M = b - a
                k = 1.5
                ax.set_ylim([a - k * M, b + k * M])

    update_labels(axes, metric_dict)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pivot_plot(df, path, metrics, cat_key, hue_key, x_key, style_key=None, ylims=dict(), log_rules=dict(),
               metric_dict=dict(), **kwargs):
    """make grid of pointplot [metric x dataset], where each point plot is [aux_key vs. metric] (e.g. iw vs. avg log_snr) """
    color_palette = sns.color_palette()
    categories = df[cat_key].unique()
    ncols = len(categories)
    nrows = len(metrics)

    if style_key is not None:
        main_keys = list(df[hue_key].unique())
        style_keys = list(df[style_key].unique())
        key_name = f"{hue_key}-{style_key}"
        df[key_name] = [f"{x}-{y}" for (x, y) in zip(df[hue_key].values, df[style_key].values)]
        df = df.drop(hue_key, 1)
        df = df.drop(style_key, 1)

        hue_order, linestyles, palette = [], [], []
        for y, _linestyle in zip(style_keys, line_styles):
            for x, hue in zip(main_keys, color_palette):
                hue_order += [f"{x}-{y}"]
                linestyles += [_linestyle]
                palette += [hue]

    else:
        key_name = hue_key
        hue_order = list(df[hue_key].unique())
        linestyles = [line_styles[0] for h in hue_order]
        palette = color_palette

    if nrows == 0 or ncols == 0:
        return None

    hue_order = list(df[key_name].unique())
    if len(categories) > 1:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows),
                                 sharex='col', sharey='row' if cat_key != 'dataset' else False)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(PLOT_WIDTH * nrows, PLOT_HEIGHT * 1))

    for j, cat in enumerate(categories):
        cat_data = df[df[cat_key] == cat]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j] if ncols > 1 else axes[i]

                for c, _key in enumerate(cat_data[key_name].unique()):
                    sub_data = cat_data[cat_data[key_name] == _key]
                    color = palette[c]

                    # extract mean and 90 percentiles
                    series = sub_data[[x_key, metric]].groupby(x_key).agg(['mean', percentile(5), percentile(95)])
                    series.reset_index(inplace=True)

                    # area plot for CI + mean
                    ax.fill_between(series[x_key], series[metric]['p_5'], series[metric]['p_95'], color=color,
                                    alpha=0.2)
                    ax.plot(series[x_key], series[metric]['mean'], color=palette[c], label=_key, marker=markers[c])

                ax.set_xlabel(x_key)
                ax.set_ylabel(metric)
                # set log scale
                set_log_scale(ax, metric, log_rules)

                if len(categories) > 1:
                    if i == 0:
                        ax.set_title(f"{cat_key} = {cat}")

                    if i < len(metrics) - 1:
                        ax.get_xaxis().set_visible(False)

                    if j == 0:
                        ax.set_ylabel(metric)
                    else:
                        ax.set_ylabel("")

                if i == len(metrics) - 1 and j == len(categories) - 1:
                    ax.legend(title=hue_key)


            except Exception as ex:
                print(
                    f"## FAILED. \n >> pivot plot: couldn't generate the axis `ax[{i}, {j}]` \nax : [{nrows},{ncols}] \nException:")
                print("--------------------------------------------------------------------------------")
                traceback.print_exception(type(ex), ex, ex.__traceback__)
                print("--------------------------------------------------------------------------------")
                print("\nException: ", ex, "\n")

    update_labels(axes, metric_dict)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
