import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ovis.reporting.plotting import PLOT_WIDTH, PLOT_HEIGHT, ESTIMATOR_STYLE, Legend
from ovis.reporting.style import DPI, MARKERS, set_matplotlib_style
from ovis.reporting.utils import get_outliers_boundaries


def plot_statistics(df, opt, logdir):
    set_matplotlib_style()

    param_name = {'b': "\mathbf{b}", 'tensor:b': "b", 'tensor:qlogits': "\phi"}.get(opt.key_filter, "\theta")
    if opt.draw_individual:
        metrics = ['individual-snr', 'grads-dsnr', 'individual-var', 'individual-magnitude']
    else:
        metrics = ['grads-snr', 'grads-dsnr', 'grads-variance', 'grads-magnitude']

    _snr = r"\operatorname{SNR}"
    _dsnr = r"\operatorname{DSNR}"
    _var = r"\operatorname{Var}"
    _g = r"\mathbf{g}"
    _avg = r"\frac{1}{D} \sum_i"
    _ex = r"\mathbb{E}"
    metrics_formaters = [lambda p: f"${_avg} {_snr}_i\: (K)$",
                         lambda p: f"${_avg} {_dsnr}_i\: (K)$",
                         lambda p: f"${_avg} {_var}[ g_i ]\: (K)$",
                         lambda p: f"${_avg} | {_ex} [ g_i ]  |\: (K)$"
                         ]

    noises = sorted(df['noise'].unique())
    estimators = df['estimator'].unique()
    nrows = len(noises)
    ncols = len(metrics)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * (0.5 + nrows)),
                             sharex='col',
                             sharey='col', squeeze=False, dpi=DPI)
    legend = Legend(fig)
    for n, noise in enumerate(sorted(noises)):

        noise_df = df[df['noise'] == noise]
        for k, metric in enumerate(metrics):
            ax = axes[n, k]

            # plot boundaries (gray lines)
            if k < 2:
                iws = list(sorted(df['iw'].unique()))
                _alpha = 0.3
                expected_max = [1e1 / k ** 0.5 for k in iws]
                expected_min = [1e-1 / k ** 0.5 for k in iws]
                ax.loglog(iws, expected_min, ":", color="#333333", basex=10, basey=10, alpha=_alpha)
                ax.loglog(iws, expected_max, ":", color="#333333", basex=10, basey=10, alpha=_alpha)

                expected_max = [1e1 * k ** 0.5 for k in iws]
                expected_min = [1e-1 * k ** 0.5 for k in iws]
                ax.loglog(iws, expected_min, ":", color="#666666", basex=10, basey=10, alpha=_alpha)
                ax.loglog(iws, expected_max, ":", color="#666666", basex=10, basey=10, alpha=_alpha)

            for e, estimator_id in enumerate(estimators):
                sub_df = noise_df[noise_df['estimator'] == estimator_id]
                iws = sub_df['iw'].values

                # color and marker
                if estimator_id in ESTIMATOR_STYLE:
                    color = ESTIMATOR_STYLE[estimator_id]['color']
                    marker = ESTIMATOR_STYLE[estimator_id]['marker']
                else:
                    color = sns.color_palette()[e]
                    marker = MARKERS[e]

                # style
                if "individual-" in metric:
                    _metrics = [m for m in sub_df.keys() if metric in m]
                    kwargs = {'alpha': 0.5, 'color': color}
                else:
                    _metrics = [metric]
                    kwargs = {'alpha': 0.9, 'color': color, 'marker': marker}

                # plot
                for i, _metric in enumerate(_metrics):
                    values = sub_df[_metric].values
                    _label = estimator_id if i == 0 else None
                    ax.loglog(iws, values, label=_label, basex=10, basey=10, **kwargs)


            # y axis scaling
            y_min, y_max = np.log10(noise_df[_metrics].values.min()), np.log10(noise_df[_metrics].values.max())
            height = y_max - y_min
            margin = 0.1
            ax.set_ylim([10**(y_min - margin * height), 10**(y_max + margin * height)])

            if n == nrows - 1:
                pass  # ax.set_xlabel("") #ax.set_xlabel("$K$")
            else:
                ax.set_xlabel("")
                ax.set_xticks([])

            if n == 0:
                ax.set_title(metrics_formaters[k](param_name))

            if k == 0:
                ax.set_ylabel(f"$\epsilon = {noise}$")
            else:
                ax.set_ylabel(f"")

            legend.update(ax)

    legend.draw()
    plt.savefig(os.path.join(logdir, "asymptotic-gradients.png"))
    plt.close()


def plot_gradients_distribution(grads_data, logdir, all_iws=False):
    """
    Plot and log the distribution of the gradients.

    :param grads_data: pd.DataFrame containing the gradients data
    :param logdir: logging directory
    :param all_iws: draw for all number of particles or plot only for [min, median, max] values
    """

    noises = grads_data['noise'].unique()
    estimators = grads_data['estimator'].unique()
    iws = list(sorted(grads_data["iw"].unique()))

    if not all_iws:
        iws = [np.min(iws), np.median(iws), np.max(iws)]

    ncols = len(noises)
    nrows = len(iws)

    if ncols > 1:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * (nrows + 0.5)), sharex='col', dpi=DPI)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=nrows,
                                 figsize=(PLOT_WIDTH * nrows, PLOT_HEIGHT * 1.5), sharex=False, sharey='row', dpi=DPI)

    legend = Legend(fig)
    for j, noise in enumerate(noises):
        grads_p = grads_data[grads_data["noise"] == noise]

        # filter data
        values = grads_p["grad"].values
        k_out = 3.0
        a, b = np.percentile(values, [25, 75])
        y_b = b + k_out * (b - a)
        y_a = a - k_out * (b - a)
        filtered_grads_p = grads_p[(grads_p["grad"] <= y_b) & (grads_p["grad"] >= y_a)]

        bars = []
        for i, iw in enumerate(iws):

            grads_p_iw = grads_p[grads_p["iw"] == iw]
            filtered_grads_p_iw = filtered_grads_p[filtered_grads_p["iw"] == iw]

            ax = axes[i, j] if ncols > 1 else axes[i]
            ax.axvline(x=0, color="gray", alpha=0.9, linestyle="-")

            for e, estimator_id in enumerate(estimators):

                # color
                if estimator_id in ESTIMATOR_STYLE:
                    color = ESTIMATOR_STYLE[estimator_id]['color']
                else:
                    color = sns.color_palette()[e]

                # plot
                filtered_data = filtered_grads_p_iw[filtered_grads_p_iw["estimator"] == estimator_id]['grad'].values
                g = sns.distplot(filtered_data, ax=ax, label=estimator_id, color=color, rug=False, kde=True, bins=None,
                                 hist_kws={"alpha": 0.3}, kde_kws={"alpha": 0.8})

                # access bar heights and store for axis scaling
                bars += [h.get_height() for h in g.patches]

                # draw mean value
                raw_data = grads_p_iw[grads_p_iw["estimator"] == estimator_id]['grad'].values
                _mean = np.mean(raw_data)
                if _mean <= y_b and _mean >= y_a:
                    ax.axvline(x=_mean, color=color, alpha=1, linestyle=":")

            if ncols > 1:
                if i == 0:
                    ax.set_title(f"$\epsilon = {noise}$")

                if i < len(iws) - 1:
                    ax.get_xaxis().set_visible(False)

                if j == 0:
                    ax.set_ylabel(f"K = {int(iw)}")
                else:
                    ax.set_ylabel("")

                # remove y axis
                ax.set_yticks([])

            else:
                ax.set_title(f"$K = {int(iw)}$")
                if i > 0:
                    ax.set_ylabel("")
                    ax.set_yticks([])

            ax.legend().remove()
            legend.update(ax)

        # set y lims
        _, m = get_outliers_boundaries(bars)
        for i, iw in enumerate(iws):
            ax = axes[i, j] if ncols > 1 else axes[i]
            ax.set_ylim([-0.1 * m, m])

    legend.draw()
    plt.savefig(os.path.join(logdir, "asymptotic-gradients-dist.png"))
    plt.close()
