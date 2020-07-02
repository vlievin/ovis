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
