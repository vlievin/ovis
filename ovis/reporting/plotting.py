import traceback
import warnings
from collections import Counter, defaultdict
from copy import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

from ovis.reporting.style import ESTIMATOR_STYLE, PLOT_WIDTH, PLOT_HEIGHT, STEP_FORMAT, ESTIMATOR_DISPLAY_NAME, \
    ESTIMATOR_ORDER, \
    ESTIMATOR_GROUPS, DPI, PLOT_TOTAL_WIDTH
from ovis.reporting.style import MARKERS, DASH_STYLES, LINE_STYLES
from ovis.reporting.utils import plot_cis
from .utils import get_outliers_boundaries


class Legend():
    """A legend handler for subplots (draws a single legend on top of the subplots)"""

    def __init__(self, figure):
        self.figure = figure
        self.legend_infos = []

    def update(self, ax):
        """get legend info from the current axis"""
        self.legend_infos += list(zip(*ax.get_legend_handles_labels()))

    def update_from_infos(self, infos):
        """get legend info from the current axis"""
        self.legend_infos += list(infos)

    def reset_linestyles(self):
        self.legend_infos = [(copy(handle), label) for handle, label in self.legend_infos]
        [handle.set_linestyle("-") for handle, label in self.legend_infos]

    def draw(self, group=False, alpha=1, insert_labels=False):
        # create legend
        legend_infos = {label: handle for handle, label in self.legend_infos}
        if len(legend_infos) == 0:
            return

        labels, handles = zip(*legend_infos.items())

        # legend for linestyles
        style_labels, style_handles = [], []

        # special case (when using seaborn)
        if 'estimator' == labels[0]:
            labels, handles = labels[1:], handles[1:]

        # parse `style` labels (when using seaborn)
        for special_key in ['iw', 'warmup', 'alpha_max', 'alpha']:
            if special_key in labels:
                q = labels.index(special_key)
                style_labels, style_handles = labels[q + 1:], handles[q + 1:]
                labels, handles = labels[:q], handles[:q]

                if special_key == 'iw':
                    # update iw label name "x" -> "K=x"
                    style_labels = [f"K={l}" for l in style_labels]
                if special_key == 'warmup':
                    _f = lambda l: eval(l) if isinstance(l, str) else l
                    style_labels = ["warmup" if _f(l) > 0 else "no warmup" for l in style_labels]
                if special_key == 'alpha_max':
                    style_labels = [r"$\alpha_{\operatorname{init}}=" + f"{l}$" for l in style_labels]
                if special_key == 'alpha':
                    style_labels = [r"$\alpha=" + f"{l}$" for l in style_labels]

        # sort labels
        if all([l in ESTIMATOR_ORDER for l in labels]):
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: ESTIMATOR_ORDER.index(x[0])))
            groups = [ESTIMATOR_GROUPS[l] for l in labels]
        else:
            groups = [0 for _ in labels]
            warnings.warn(f"Not all labels = `{labels}` are in `ESTIMATOR_ORDER`, the legend won't be sorted")

        # group number for the styles
        style_groups = [max(groups) + 1 for l in style_labels]

        # format names
        labels = [ESTIMATOR_DISPLAY_NAME.get(l, l) for l in labels]

        # define number of columns
        all_groups = groups + style_groups

        # grouping
        if group and len(set(all_groups)) > 1:
            ncol = len(set(all_groups))

            # get longer group
            most_common_key, max_length = Counter(all_groups).most_common(1)[0]
            _all_labels = list(labels) + list(style_labels)
            _all_handles = list(handles) + list(style_handles)

            # build dicitonaruy {group : [labels]}
            all_labels, all_handles = defaultdict(list), defaultdict(list)
            for g, l, h in zip(all_groups, _all_labels, _all_handles):
                all_labels[g] += [l]
                all_handles[g] += [h]

            # extend all groups to match max length
            _null_patch = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                                                       visible=False)
            for g in all_groups:
                all_labels[g] = all_labels[g] + (max_length - len(all_labels[g])) * [""]
                all_handles[g] = all_handles[g] + (max_length - len(all_handles[g])) * [_null_patch]

            # flatten everything
            all_labels = [l for ls in all_labels.values() for l in ls]
            all_handles = [l for ls in all_handles.values() for l in ls]

            # number of rows
            nrow = max_length

        else:
            # append style legend
            all_labels = list(labels) + list(style_labels)
            all_handles = list(handles) + list(style_handles)
            ncol = len(labels)
            nrow = 1

        # infer sizes
        width, height = self.figure.get_size_inches()
        legend_base_size = 0.5
        legend_row_size = 0.3
        margin = 0.5
        legend_height = legend_base_size + nrow * legend_row_size

        # set tight layout and add margin at the top for the legend
        plt.tight_layout()
        self.figure.subplots_adjust(top=1 - (legend_height + margin) / height)

        # draw the legend
        legend = self.figure.legend(handles=all_handles, labels=all_labels, ncol=ncol, loc='lower center',
                                    bbox_to_anchor=(0, 1 - (legend_height) / height, 1, 1), fancybox=False,
                                    shadow=False, fontsize='medium' if ncol < 8 else 'x-small')

        if alpha is not None:
            for l in legend.get_lines():
                l.set_alpha(1)


def set_log_scale(ax, label, log_rules, axis='y'):
    """use log scale for the axis `ax` depending on the `log_rules` dictionary"""
    if ':' in label:
        label = label.split(':')[-1]
    rule = log_rules.get(label, 'linear')
    if rule == 'linear':
        pass
    elif rule == 'log':
        if axis == 'y':
            ax.set_yscale('log')
        elif axis == 'x':
            ax.set_xscale('log')
        else:
            raise ValueError(f"Unknown axis type `{axis}`")
    else:
        raise ValueError(f"Unknown log rule `{rule}`")


def update_labels(axes, metric_dict, agg_fns=dict()):
    """
    update the axes labels based on the `metric_dict` dictionary and optional `agg_fns`
    """

    def _parse(label):
        return label.split(':')[-1]

    for ax in axes.reshape(-1):

        xlabel = _parse(ax.get_xlabel())
        ylabel = _parse(ax.get_ylabel())

        if xlabel in metric_dict.keys():
            label = metric_dict[xlabel]
            ax.set_xlabel(label)

        if ylabel in metric_dict.keys():
            label = metric_dict[ylabel]
            _label = ax.get_ylabel()
            # append `^{header}$` to `$\mathcal{L}`

            for k in ['loss/L_k', 'loss/elbo', 'loss/kl']:
                if k in ylabel:
                    if 'train:' in _label:
                        label = label[:-1] + "^{\operatorname{train}}$"
                    elif 'valid:' in _label:
                        label = label[:-1] + "^{\operatorname{valid}}$"
                    elif 'test:' in _label:
                        label = label[:-1] + "^{\operatorname{test}}$"

            if len(agg_fns):
                if _label in agg_fns.keys():
                    agg_label = {'last': "", "mean": "train. avg. "}.get(agg_fns[_label], f"{agg_fns[_label]}. ")
                    label = f"{agg_label}{label}"

            ax.set_ylabel(label)


def basic_curves_plot(logs, path, metrics, main_key, style_key=None, ylims=dict(), log_rules=dict(),
                      metric_dict=dict(), ncols=2, **kwargs):
    """
    Make a grid of line plots with std intervals. Each subplot correspond to one metric.
    :param path: output file
    :param logs: DataFrame containing all the training data per time step
    :param metrics: metrics to plot (elbo, nll, ...)
    :param main_key: key to used for coloring
    :param style_key: key used for styling (Optional)
    :param ylims: dictionary for the y axis boundaries (Optional)
    :param metric_dict: dictionary for the axes names [Optional]
    :param ncols: number of columns
    :return: None
    """

    N = len(metrics)
    nrows = -(-N // ncols)  # ceiling division
    hue_order = list(logs[main_key].unique())
    fig, axes = plt.subplots(nrows, ncols, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * (nrows + 0.5)), dpi=DPI)
    legend = Legend(fig)
    for i, k in tqdm(list(enumerate(metrics)), desc="|    subplots"):
        u = i // ncols
        v = i % ncols
        ax = axes[u, v]

        if main_key == 'estimator':
            palette = [ESTIMATOR_STYLE[h_key]['color'] for h_key in hue_order]
        else:
            palette = sns.color_palette()

        sns.lineplot(x="step", y="_value",
                     hue=main_key,
                     hue_order=hue_order,
                     style=style_key,
                     data=logs[logs['_key'] == k],
                     ax=ax,
                     dashes=DASH_STYLES,
                     palette=palette,
                     )

        # set log scale and set the axes labels
        set_log_scale(ax, k, log_rules, axis='y')
        ax.set_ylabel(k)

        # y limits: using the known values or using outlier boundaries
        if k in ylims:
            ax.set_ylim(ylims[k])
        elif ax.get_yaxis().get_scale() != 'log':
            ys = logs[logs['_key'] == k]['_value'].values.tolist()
            if len(ys):
                ax.set_ylim(get_outliers_boundaries(ys, k=1.5))

        # update legend and format x axis ticks
        ax.get_legend().remove()
        legend.update(ax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(STEP_FORMAT))

    # update the axes names and draw the legend
    update_labels(axes, metric_dict)
    legend.draw(group=True)
    plt.savefig(path)
    plt.close()


def detailed_curves_plot(logs, path, metrics, hue_key, auxiliary_key, style_key=None, ylims=dict(), log_rules=dict(),
                         metric_dict=dict(), scale_y=True, **kwargs):
    """
    Make a grid of plots where each row corresponds to a `metric` and each column to an `auxiliary_key`.
    The hue is based on the `hue_key` and the linestyle [Optional] based on the `style_key`.
    :param logs: logs data (DataFrame)
    :param path: output path
    :param metrics: list of metrics to draw
    :param hue_key: color key
    :param auxiliary_key: column key
    :param style_key: linestyle key
    :param ylims: custom y axis limits
    :param log_rules: custom log axes rules [Optional]
    :param metric_dict: dictionary for the axes names [Optional]
    :param scale_y: scale/don't scale y axis
    """
    aux_keys = logs[auxiliary_key].unique()
    nrows = len(metrics)
    ncols = len(aux_keys)

    if nrows == 0 or ncols == 0:
        return None

    hue_order = list(logs[hue_key].unique())
    _factor = 3 / 4
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(_factor * PLOT_WIDTH * ncols, _factor * PLOT_HEIGHT * (nrows + 0.5)),
                             sharex='col',
                             sharey='row', squeeze=False, dpi=DPI)
    legend = Legend(fig)
    for j, aux_key in tqdm(list(enumerate(sorted(aux_keys))), desc="|  aux. keys"):
        aux_data = logs[logs[auxiliary_key] == aux_key]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j] if ncols > 1 else axes[i]
                data = aux_data[aux_data["_key"] == metric]

                if hue_key == 'estimator':
                    palette = [ESTIMATOR_STYLE[h_key]['color'] for h_key in hue_order]
                else:
                    palette = sns.color_palette()

                sns.lineplot(x="step", y="_value",
                             hue=hue_key,
                             hue_order=hue_order,
                             style=style_key,
                             data=data,
                             ax=ax,
                             palette=palette,
                             alpha=0.8
                             )

                # set log scale
                set_log_scale(ax, metric, log_rules, axis='y')

                # define axis labels and hide x,y axis in the middle plots
                if i == 0:
                    _name = {'iw': r'$K$', 'gamma': r'$\beta$', 'gamma_min': r'$\beta_{\operatorname{min}}$'}.get(
                        auxiliary_key, auxiliary_key)
                    ax.set_title(f"{_name} = {aux_key}")

                if i < len(metrics) - 1:
                    # ax.set_xticklabels([])
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_ylabel(metric)
                else:
                    ax.set_ylabel("")
                    # ax.set_yticklabels([])

                ax.get_legend().remove()
                legend.update(ax)
                ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(STEP_FORMAT))

            except:
                warnings.warn(f">> detailed plot: couldn't generate the axis `ax[{i}, {j}]`")

    # scale y axes
    if scale_y:
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            data = logs[logs["_key"] == metric]
            if metric in ylims:
                ax.set_ylim(ylims[metric])
            elif ax.get_yaxis().get_scale() != 'log':
                ys = data['_value'].values.tolist()
                if len(ys):
                    a, b = np.percentile(ys, [25, 75])
                    M = b - a
                    k = 1.5
                    ax.set_ylim([a - k * M, b + k * M])

    update_labels(axes, metric_dict)
    legend.draw(group=True)

    plt.savefig(path)
    plt.close()


def pivot_plot(data, path, metrics, category_key, hue_key, x_key, style_key=None, log_rules=dict(),
               metric_dict=dict(), agg_fns=[], **kwargs):
    """
    Draw a grid of plots `metric(x_key)` where each row is a `metric` and each column a `category`.
    If there is only one `category`, then all `metrics` are displayed on one row.
    :param data: DataFrame
    :param path: output path
    :param metrics: list of metrics to draw
    :param category_key: list of categories (e.g. dataset, depth, ...)
    :param hue_key: color key
    :param x_key: x axis key
    :param style_key: linestyle key
    :param log_rules: custom log axes rules [Optional]
    :param metric_dict: dictionary for the axes names [Optional]
    :param agg_fns: dictionary containing the `pivot_metrics_agg_ids`
    """

    data = data.dropna()
    categories = data[category_key].unique()
    ncols = len(categories)
    nrows = len(metrics)

    if nrows == 0 or ncols == 0:
        warnings.warn(f"n. categories = `{len(categories)}` and n. metrics = `{len(metrics)}`: nothing to draw.")
        return None

    hue_index = {l: i for i, l in enumerate(sorted(data[hue_key].unique()))}
    style_order = list(sorted(list(data[style_key].unique()))) if style_key is not None else None
    legend_height = 0.5 if style_order is None else 0.3 + 0.2 * len(style_order)
    if len(categories) > 1:
        width = PLOT_TOTAL_WIDTH if PLOT_TOTAL_WIDTH is not None else PLOT_WIDTH * ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, PLOT_HEIGHT * (nrows + legend_height)),
                                 sharex='col', sharey='row' if category_key != 'dataset' else False, dpi=DPI)
    else:
        width = PLOT_TOTAL_WIDTH if PLOT_TOTAL_WIDTH is not None else PLOT_WIDTH * nrows
        fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(width, 1 * PLOT_HEIGHT * (1 + legend_height)), dpi=DPI)

    legend = Legend(fig)
    for j, cat in enumerate(categories):
        cat_data = data[data[category_key] == cat]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j] if ncols > 1 else axes[i]

                for c, h_key in enumerate(cat_data[hue_key].unique()):
                    sub_data = cat_data[cat_data[hue_key] == h_key]

                    # retrieve `style` from known `ESTIMATOR_STYLE` if available
                    if hue_key == 'estimator':
                        style = copy(ESTIMATOR_STYLE[h_key])
                        style.pop('linestyle')
                    else:
                        idx = hue_index[h_key]
                        style = {'color': sns.color_palette()[idx], 'marker': MARKERS[idx]}

                    _styles = sub_data[style_key].unique() if style_key is not None else [None]
                    for s, s_key in enumerate(_styles):
                        if s_key is not None:
                            sub_sub_data = sub_data[sub_data[style_key] == s_key]
                            style['linestyle'] = LINE_STYLES[style_order.index(s_key)]
                        else:
                            sub_sub_data = sub_data

                        # extract mean and std
                        series = sub_sub_data[[x_key, metric]].groupby(x_key).agg(['mean', 'std'])
                        series.reset_index(inplace=True)

                        # area plot for CI
                        ax.fill_between(series[x_key], series[metric]['mean'] - 0.5 * series[metric]['std'],
                                        series[metric]['mean'] + 0.5 * series[metric]['std'], color=style['color'],
                                        alpha=0.2)

                        # plot CI ticks
                        capsize = 0  # issue with log-scale
                        alpha = 0.75
                        if capsize > 0:
                            ci_low = series[metric]['mean'] - 0.5 * series[metric]['std']
                            ci_high = series[metric]['mean'] + 0.5 * series[metric]['std']
                            plot_cis(ax, series[x_key], ci_low, ci_high, capsize=capsize, color=style['color'],
                                     alpha=alpha)

                        # plot mean value
                        ax.plot(series[x_key], series[metric]['mean'], label=h_key, markersize=0, alpha=alpha, **style)
                        ax.plot(series[x_key], series[metric]['mean'], label=h_key, alpha=1, **style)

                # define labels and log axis scales
                ax.set_xlabel(x_key)
                ax.set_ylabel(metric)
                set_log_scale(ax, x_key, log_rules, axis='x')
                set_log_scale(ax, metric, log_rules, axis='y')

                # set titles for each column and set axis names
                if len(categories) > 1:
                    if i == 0:
                        ax.set_title(f"{category_key} = {cat}")

                    if i < len(metrics) - 1:
                        ax.get_xaxis().set_visible(False)

                    if j == 0:
                        ax.set_ylabel(metric)
                    else:
                        ax.set_ylabel("")

                legend.update(ax)

            except Exception as ex:
                print(
                    f"## FAILED. \n >> pivot plot: couldn't generate the axis `ax[{i}, {j}]` \nax : [{nrows},{ncols}] \nException:")
                print("--------------------------------------------------------------------------------")
                traceback.print_exception(type(ex), ex, ex.__traceback__)
                print("--------------------------------------------------------------------------------")
                print("\nException: ", ex, "\n")

    # update legend with style labels
    if style_order is not None and len(style_order) > 1:
        legend.reset_linestyles()
        legend.update_from_infos([(None, style_key)])
        for k, label in enumerate(style_order):
            linestyle = LINE_STYLES[k]
            patch = Line2D([0], [0], color="black", linestyle=linestyle)
            legend.update_from_infos([(patch, label)])

    # update axis labels and draw the legend
    update_labels(axes, metric_dict, agg_fns=agg_fns)
    legend.draw(group=style_order is not None and len(style_order) > 1)

    plt.savefig(path)
    plt.close()
