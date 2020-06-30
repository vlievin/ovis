import numpy as np


def percentile(n):
    def percentile_fn(x):
        return np.percentile(x, n)

    percentile_fn.__name__ = 'p_%s' % n
    return percentile_fn


def get_outliers_boundaries(values, k=1.5):
    a, b = np.percentile(values, [25, 75])
    return [a - k * (b - a), b + k * (b - a)]


def plot_cis(ax, at, ci_low, ci_high, color, capsize, **kws):
    ax.plot([at, at], [ci_low, ci_high], color=color, **kws)
    if capsize is not None:
        ax.plot([at - capsize / 2, at + capsize / 2],
                [ci_low, ci_low], color=color, **kws)
        ax.plot([at - capsize / 2, at + capsize / 2],
                [ci_high, ci_high], color=color, **kws)