import numpy as np

from .style import ESTIMATOR_ORDER


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


def smooth(x, window_len=5, window='hanning'):
    """
    From: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    pad = (window_len - 1) // 2
    y = y[pad:-pad]
    return y


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


def sort_estimator_keys(keys):
    keys.sort(key=lambda x: ESTIMATOR_ORDER.index(x) if x in ESTIMATOR_ORDER else -1, reverse=True)


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
