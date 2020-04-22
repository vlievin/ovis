import os

import numpy as np

from .config import *
from .gradients import *
from .plotting import PLOT_WIDTH, PLOT_HEIGHT, markers

_sep = os.get_terminal_size().columns * "-"

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette()


def get_outliers_boundaries(values, k=1.5):
    a, b = np.percentile(values, [25, 75])
    return [a - k * (b - a), b + k * (b - a)]


def gen_dataset(model, npoints):
    # generate the dataset
    model.mu.data = torch.randn_like(model.mu.data)
    return model.sample_from_prior(N=npoints)['px'].sample()


def set_optimal_parameters(model, x):
    mu = x.mean(dim=0, keepdim=True).data
    model.mu.data = mu.data.view_as(model.mu.data)  # mu^*
    model.A.data = 0.5 * torch.eye(x.shape[1], device=x.device).view_as(model.A.data)  # A = I / 2
    model.b.data = 0.5 * mu.view_as(model.b.data)  # b = 0.5 mu^*


def perturbate_weights(model, noise_scale):
    model.mu.data += noise_scale * torch.randn_like(model.mu.data)
    model.A.data += noise_scale * torch.randn_like(model.A.data)
    model.b.data += noise_scale * torch.randn_like(model.b.data)


def evaluate(estimator, model, x, config, seed, base_logger, desc):
    print(_sep)
    torch.manual_seed(seed)
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    base_logger.info(
        f"{desc} | L_{estimator.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, KL = {diagnostics['loss']['kl'].mean().item():.6f}, NLL = {diagnostics['loss']['nll'].mean().item():.6f}")

    return diagnostics


def compute_true_grads(estimator, model, x, mc_samples, seed=None, **kwargs):
    seed = seed + 1 if seed is not None else None  # make sure that the true grads is computed with a different seed to avoid spurious corelations
    _, meta = get_batch_gradients_statistics(estimator, model, x, n_samples=mc_samples, return_grads=True, seed=seed,
                                             **kwargs)
    true_grads = meta['grads'].mean(dim=0)

    # mu = x.mean(dim=0).data
    # true_grads = 0.5 * mu - model.b.data

    return true_grads / true_grads.norm(p=2, dim=-1)


def get_estimator(estimator_id, iw):
    if "-mc" in estimator_id:
        Estimator, config = get_config(estimator_id.replace("-mc", ""))
        _mc, _iw = iw, 1
    else:
        Estimator, config = get_config(estimator_id)
        _mc, _iw = 1, iw

    return Estimator(baseline=None, mc=_mc, iw=_iw), config


def log_grads_data(analysis_data, base_logger, estimator_id, iw):
    grad_data = analysis_data.get('grads', {})
    base_logger.info(
        f"{estimator_id}, iw = {iw} | snr = {grad_data.get('snr', 0.):.3E}, dsnr = {grad_data.get('dsnr', 0.):.3E}, variance = {grad_data.get('variance', 0.):.3E}, magnitude = {grad_data.get('magnitude', 0.):.3E}, dir = {grad_data.get('direction', 0.):.3E}")
    snr_data = analysis_data.get('snr', {})
    base_logger.info(
        f"{estimator_id}, iw = {iw} | snr | p5 = {snr_data.get('p5', 0.):.3E}, p25 = {snr_data.get('p25', 0.):.3E}, p50 = {snr_data.get('p50', 0.):.3E}, p75 = {snr_data.get('p75', 0.):.3E}, p95 = {snr_data.get('p95', 0.):.3E}")


def plot_statistics(df, opt, logdir):
    # plotting
    param_name = {'tensor:b': "b", 'tensor:qlogits': "\phi"}.get(opt.key_filter, "\theta")
    if opt.draw_individual:
        metrics = ['individual-snr', 'grads-dsnr', 'individual-var', 'individual-magnitude', 'grads-direction']
    else:
        metrics = ['grads-snr', 'grads-dsnr', 'grads-variance', 'grads-magnitude', 'grads-direction']

    _true_grads_name = "\Delta_{" + f"{opt.iw_oracle}" + "}^{"+ f"{str(opt.oracle).replace('pathwise-', '')}" + "}"
    metrics_formaters = [lambda p: f"$SNR_K({param_name}) $",
                         lambda p: f"$DSNR_K({param_name}) $",
                         lambda p: f"$Var \Delta_K({param_name}) $",
                         lambda p: f"$| \Delta_K({param_name}) |$",
                         lambda p: f"$cosine( \Delta_K({param_name}), E[ {_true_grads_name}({param_name}) ])$",
                         ]

    noises = sorted(df['noise'].unique())
    estimators = df['estimator'].unique()
    nrows = len(noises)
    ncols = len(metrics)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows), sharex='col',
                             sharey='col', squeeze=False)
    for n, noise in enumerate(sorted(noises)):

        noise_df = df[df['noise'] == noise]
        for k, metric in enumerate(metrics):
            ax = axes[n, k]

            if k == 0:
                iws = list(sorted(df['iw'].unique()))
                expected_max = [1e1 / k ** 0.5 for k in iws]
                expected_min = [1e-1 / k ** 0.5 for k in iws]
                ax.loglog(iws, expected_min, ":", color="#333333", basex=10, basey=10)
                ax.loglog(iws, expected_max, ":", color="#333333", basex=10, basey=10)

            for e, estimator_id in enumerate(estimators):
                sub_df = noise_df[noise_df['estimator'] == estimator_id]
                iws = sub_df['iw'].values

                if "individual-" in metric:
                    _metrics = [m for m in sub_df.keys() if metric in m]
                    kwargs = {'alpha': 0.5, 'color': colors[e]}
                else:
                    _metrics = [metric]
                    kwargs = {'alpha': 0.9, 'color': colors[e], 'marker': markers[e], 'markersize': 6}

                for i, _metric in enumerate(_metrics):

                    values = sub_df[_metric].values
                    _label = estimator_id if i == 0 else None
                    if "direction" in _metric:
                        ax.plot(iws, values, label=_label, **kwargs)
                        ax.set_xscale('log')
                    else:
                        ax.loglog(iws, values, label=_label, basex=10, basey=10, **kwargs)

            if n == nrows - 1:
                ax.set_xlabel("$K$")
            else:
                ax.set_xlabel("")
                ax.set_xticks([])

            if n == 0:
                ax.set_title(metrics_formaters[k](param_name))

            if k == 0:
                ax.set_ylabel(f"$\epsilon = {noise}$")
            else:
                ax.set_ylabel(f"")

            if k == len(metrics) - 1 and n == len(noises) - 1:
                ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "gradients.png"))
    plt.close()


def plot_gradients_distribution(grads, logdir):
    def agg(s):
        return f"{np.mean(s):.3f} +/- {np.std(s):.3f} (n={len(s)})"

    print(_sep)
    print("Gradients")
    print(_sep)
    print(grads.pivot_table(columns="estimator", index=["noise", "iw"], values="grad", aggfunc=agg))
    print(_sep)

    noises = grads['noise'].unique()
    estimators = grads['estimator'].unique()
    iws = list(sorted(grads["iw"].unique()))

    ncols = len(noises)
    nrows = len(iws)

    if ncols > 1:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows), sharex='col')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=nrows,
                                 figsize=(PLOT_WIDTH * nrows, PLOT_HEIGHT * 1), sharex='row', sharey='row')

    for j, noise in enumerate(noises):
        grads_p = grads[grads["noise"] == noise]

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
                color = colors[e]

                filtered_data = filtered_grads_p_iw[filtered_grads_p_iw["estimator"] == estimator_id]['grad'].values
                g = sns.distplot(filtered_data, ax=ax, label=estimator_id, color=color, rug=False, kde=True, bins=64,
                                 hist_kws={"alpha": 0.3}, kde_kws={"alpha": 0.8})

                # access bar heights
                bars += [h.get_height() for h in g.patches]

                # draw mean value
                raw_data = grads_p_iw[grads_p_iw["estimator"] == estimator_id]['grad'].values
                _mean = np.mean(raw_data)
                if _mean <= y_b and _mean >= y_a:
                    ax.axvline(x=_mean, color=color, alpha=1, linestyle="--")

            if ncols > 1:
                if i == 0:
                    ax.set_title(f"$\epsilon = {noise}$")

                if i < len(iws) - 1:
                    ax.get_xaxis().set_visible(False)

                if j == 0:
                    ax.set_ylabel(f"K = {iw}")
                else:
                    ax.set_ylabel("")

                # remove y axis
                ax.set_yticks([])

            else:
                ax.set_title(f"$K = {iw}$")
                if i > 0:
                    ax.set_ylabel("")
                    ax.set_yticks([])

            if not (i == nrows - 1 and j == ncols - 1):
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

        # set y lims
        _, m = get_outliers_boundaries(bars)
        for i, iw in enumerate(iws):
            ax = axes[i, j] if ncols > 1 else axes[i]
            ax.set_ylim([-0.1 * m, m])

    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(logdir, "gradients-dist.png"))
    plt.close()
