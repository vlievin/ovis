import traceback
import warnings
from collections import Counter, defaultdict
from copy import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from tqdm import tqdm

from lib.style import MARKERS, DASH_STYLES, LINE_STYLES
from .style import ESTIMATOR_STYLE, PLOT_WIDTH, PLOT_HEIGHT, STEP_FORMAT, ESTIMATOR_DISPLAY_NAME, ESTIMATOR_ORDER, \
    ESTIMATOR_GROUPS, DPI, PLOT_TOTAL_WIDTH


class Legend():
    """A small class to plot one legend for all subplots"""

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

        # special case
        if 'estimator' == labels[0]:
            labels, handles = labels[1:], handles[1:]

        for special_key in ['iw', 'warmup', 'gamma_min']:
            if special_key in labels:
                q = labels.index(special_key)
                style_labels, style_handles = labels[q + 1:], handles[q + 1:]
                labels, handles = labels[:q], handles[:q]

                if special_key == 'iw':
                    # update iw label name "x" -> "K=x"
                    style_labels = [f"K={l}" for l in style_labels]
                if special_key == 'warmup':
                    _f = lambda l : eval(l) if isinstance(l, str) else l
                    style_labels = ["warmup" if _f(l) > 0 else "no warmup" for l in style_labels]
                if special_key == 'gamma_min':
                    style_labels = [r"$1- \alpha_{\operatorname{init}}="+f"{l}$" for l in style_labels]

        # sort names
        if all([l in ESTIMATOR_ORDER for l in labels]):
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: ESTIMATOR_ORDER.index(x[0])))
            groups = [ESTIMATOR_GROUPS[l] for l in labels]
        else:
            groups = [0 for _ in labels]
            print(">>> legend will not be sorted")
            print(labels)

        # group number for the styles
        style_groups = [max(groups) + 1 for l in style_labels]

        # format names
        labels = [ESTIMATOR_DISPLAY_NAME.get(l, l) for l in labels]

        # define number of columns
        all_groups = groups + style_groups

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
        self.figure.subplots_adjust(top= 1 - (legend_height + margin) / height)
        legend = self.figure.legend(handles=all_handles, labels=all_labels, ncol=ncol, loc='lower center',
                           bbox_to_anchor=(0,  1 - (legend_height) / height, 1, 1) ,fancybox=False, shadow=False) #,  fontsize='x-small') #, title=hue_key

        if alpha is not None:
            for l in legend.get_lines():
                l.set_alpha(1)


def set_log_scale(ax, label, log_rules, axis='y'):
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
                    agg_label = {'last' : "", "mean": "train. avg. "}.get(agg_fns[_label], f"{agg_fns[_label]}. ")
                    label = f"{agg_label}{label}"

            ax.set_ylabel(label)


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

        # set log scale
        set_log_scale(ax, k, log_rules, axis='y')

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

        ax.get_legend().remove()
        legend.update(ax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(STEP_FORMAT))
    # draw legend in the last plot
    # patches = [mpatches.Patch(color=sns.color_palette()[i], label=key) for i, key in enumerate(hue_order)]
    # axes[nrows - 1, ncols - 1].legend(handles=patches)

    update_labels(axes, metric_dict)

    legend.draw(group=True)
    plt.savefig(path)
    plt.close()


def plot_cis(ax, at, ci_low, ci_high, color, capsize, **kws):
    ax.plot([at, at], [ci_low, ci_high], color=color, **kws)
    if capsize is not None:
        ax.plot([at - capsize / 2, at + capsize / 2],
                [ci_low, ci_low], color=color, **kws)
        ax.plot([at - capsize / 2, at + capsize / 2],
                [ci_high, ci_high], color=color, **kws)


def detailed_plot(logs, path, metrics, main_key, auxiliary_key, style_key=None, ylims=dict(), log_rules=dict(),
                  metric_dict=dict(), scale_y=True,**kwargs):
    """make a grid of plot metrics x aux. keys values"""
    aux_keys = logs[auxiliary_key].unique()
    nrows = len(metrics)
    ncols = len(aux_keys)

    if nrows == 0 or ncols == 0:
        return None

    hue_order = list(logs[main_key].unique())
    step_min = np.percentile(logs['step'].values.tolist(), 10) if len(logs['step']) else 0  # used to filter first steps
    _factor = 3/4
    fig, axes = plt.subplots(nrows, ncols, figsize=( _factor * PLOT_WIDTH * ncols, _factor * PLOT_HEIGHT * (nrows+0.5)), sharex='col',
                             sharey='row', squeeze=False, dpi=DPI)
    legend = Legend(fig)
    for j, aux_key in tqdm(list(enumerate(sorted(aux_keys))), desc="|  aux. keys"):
        aux_data = logs[logs[auxiliary_key] == aux_key]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j] if ncols > 1 else axes[i]
                data = aux_data[aux_data["_key"] == metric]

                if main_key == 'estimator':
                    palette = [ESTIMATOR_STYLE[h_key]['color'] for h_key in hue_order]
                else:
                    palette = sns.color_palette()

                sns.lineplot(x="step", y="_value",
                             hue=main_key,
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
                    _name = {'iw': r'$K$', 'gamma': r'$\beta$', 'gamma_min':r'$\beta_{\operatorname{min}}$' }.get(auxiliary_key, auxiliary_key)
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


            print(">>> metric", metric)
            print(ylims)

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
    legend.draw(group=True)

    plt.savefig(path)
    plt.close()


def pivot_plot(df, path, metrics, cat_key, hue_key, x_key, style_key=None, ylims=dict(), log_rules=dict(),
               metric_dict=dict(), agg_fns=[], **kwargs):
    """make grid of pointplot [metric x dataset], where each point plot is [aux_key vs. metric] (e.g. iw vs. avg log_snr) """

    df = df.dropna()

    categories = df[cat_key].unique()
    ncols = len(categories)
    nrows = len(metrics)

    # if style_key is not None:
    #     main_keys = list(df[hue_key].unique())
    #     style_keys = list(df[style_key].unique())
    #     key_name = f"{hue_key}-{style_key}"
    #     color_palette = [ESTIMATOR_STYLE[key]['color'] for key in main_keys]
    #     df[key_name] = [f"{x}-{y}" for (x, y) in zip(df[hue_key].values, df[style_key].values)]
    #     df = df.drop(hue_key, 1)
    #     df = df.drop(style_key, 1)
    #
    #     hue_order, linestyles, palette = [], [], []
    #     for y, _linestyle in zip(style_keys, LINE_STYLES):
    #         for x, hue in zip(main_keys, color_palette):
    #             hue_order += [f"{x}-{y}"]
    #             linestyles += [_linestyle]
    #             palette += [hue]
    #
    # else:
    #     key_name = hue_key
    #     hue_order = list(df[hue_key].unique())
    #     linestyle = [ESTIMATOR_STYLE[key]['linestyle'] for key in hue_order]
    #     palette = [ESTIMATOR_STYLE[key]['color'] for key in hue_order]

    if nrows == 0 or ncols == 0:
        return None

    hue_index = {l: i for i, l in enumerate(sorted(df[hue_key].unique()))}
    hue_order = df[hue_key].unique()
    style_order = list(sorted(list(df[style_key].unique()))) if style_key is not None else None
    legend_height = 0.5 if style_order is None else 0.3 + 0.2 * len(style_order)
    if len(categories) > 1:
        width = PLOT_TOTAL_WIDTH if PLOT_TOTAL_WIDTH is not None else PLOT_WIDTH * ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, PLOT_HEIGHT * (nrows+legend_height)),
                                 sharex='col', sharey='row' if cat_key != 'dataset' else False, dpi=DPI)
    else:
        width = PLOT_TOTAL_WIDTH if PLOT_TOTAL_WIDTH is not None else PLOT_WIDTH * nrows
        fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(width, 1 * PLOT_HEIGHT * (1 + legend_height) ), dpi=DPI)

    legend = Legend(fig)

    for j, cat in enumerate(categories):
        cat_data = df[df[cat_key] == cat]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j] if ncols > 1 else axes[i]

                for c, h_key in enumerate(cat_data[hue_key].unique()):

                    sub_data = cat_data[cat_data[hue_key] == h_key]

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


                        # extract mean and 90 percentiles
                        series = sub_sub_data[[x_key, metric]].groupby(x_key).agg(['mean', 'std'])
                        series.reset_index(inplace=True)

                        # area plot for CI + mean
                        ax.fill_between(series[x_key], series[metric]['mean'] - 0.5 * series[metric]['std'],
                                        series[metric]['mean'] + 0.5 * series[metric]['std'], color=style['color'],
                                        alpha=0.2)

                        ci_low = series[metric]['mean'] - 0.5 * series[metric]['std']
                        ci_high = series[metric]['mean'] + 0.5 * series[metric]['std']
                        capsize = 0
                        alpha = 0.75
                        ax.plot(series[x_key], series[metric]['mean'], label=h_key, markersize=0, alpha=alpha, **style)
                        ax.plot(series[x_key], series[metric]['mean'], label=h_key, alpha=1, **style)
                        plot_cis(ax, series[x_key], ci_low, ci_high, capsize=capsize, color=style['color'], alpha=alpha)


                # if hue_key == 'estimator':
                #     palette = [ESTIMATOR_STYLE[h_key]['color'] for h_key in hue_order]
                # else:
                #     palette = sns.color_palette()
                #
                # sns.lineplot(x=x_key, y=metric,
                #              hue=hue_key,
                #              hue_order=hue_order,
                #              style=style_key,
                #              data=cat_data,
                #              ax=ax,
                #              palette=palette,
                #              alpha=0.8,
                #              markers=True
                #              )

                # ylabel
                ax.set_xlabel(x_key)
                ax.set_ylabel(metric)
                # set log scale
                set_log_scale(ax, x_key, log_rules, axis='x')
                set_log_scale(ax, metric, log_rules, axis='y')

                # x ticks
                # locmaj = matplotlib.ticker.LogLocator(base=10.0, subs='all')
                # ax.xaxis.set_major_locator(locmaj)

                if len(categories) > 1:
                    if i == 0:
                        ax.set_title(f"{cat_key} = {cat}")

                    if i < len(metrics) - 1:
                        ax.get_xaxis().set_visible(False)

                    if j == 0:
                        ax.set_ylabel(metric)
                    else:
                        ax.set_ylabel("")

                # if i == len(metrics) - 1 and j == len(categories) - 1:
                #     ax.legend(title=hue_key)
                # ax.get_legend().remove()
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
                print(">>>> ", label, linestyle)
                patch = Line2D([0], [0], color="black", linestyle=linestyle)
                legend.update_from_infos([ (patch, label)])

    # update axis labels
    update_labels(axes, metric_dict, agg_fns=agg_fns)

    legend.draw(group=style_order is not None and len(style_order) > 1)

    plt.savefig(path)
    plt.close()



