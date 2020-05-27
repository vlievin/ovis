import seaborn as sns


def set_style():
    sns.set(style="whitegrid")
    sns.set(style="ticks")
    sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 2.4, "lines.markersize": 12})


DPI = 200
PLOT_WIDTH = 5
PLOT_HEIGHT = 3
STEP_FORMAT = '{x:.0e}'

ESTIMATOR_STYLE = {
    'copt-alpha0': {'color': "#E6C445", 'marker': ">", 'linestyle': "-"},
    'copt-alpha1': {'color': "#B88A25", 'marker': "<", 'linestyle': "--"},
    'copt-aux1': {'color': "#E89C66", 'marker': "^", 'linestyle': ":"},
    'copt-aux10': {'color': "#D15C2A", 'marker': "^", 'linestyle': "--"},
    'copt-aux50': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'copt-aux100': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'reinforce': {'color': "#7A9396", 'marker': "o", 'linestyle': "-"},
    'vimco-arithmetic': {'color': "#4BAD5D", 'marker': "h", 'linestyle': "-"},
    'vimco-geometric': {'color': "#358141", 'marker': "H", 'linestyle': "--"},
    'tvo': {'color': "#336DA2", 'marker': "P", 'linestyle': "-"},
    'ww': {'color': "#77B6D7", 'marker': "X", 'linestyle': "-"},
    'pathwise': {'color': "#968FA1", 'marker': "s", 'linestyle': "-"}
}

ESTIMATOR_ORDER = [
    'copt-aux100',
    'copt-aux50',
    'copt-aux10',
    'copt-aux1',
    'copt-alpha1',
    'copt-alpha0',
    'vimco-arithmetic',
    'vimco-geometric',
    'reinforce',
    'ww',
    'tvo',
    'pathwise'
]

ESTIMATOR_GROUPS = {
    'copt-aux100': 0,
    'copt-aux50': 0,
    'copt-aux10': 0,
    'copt-aux1': 0,
    'copt-alpha1': 1,
    'copt-alpha0': 1,
    'vimco-arithmetic': 2,
    'vimco-geometric': 2,
    'reinforce': 2,
    'ww': 3,
    'tvo': 3,
    'pathwise': 3
}

ESTIMATOR_DISPLAY_NAME = {
    'copt-alpha0': r"$c^{\alpha = 0}$",
    'copt-alpha1': r"$c^{\alpha = 1}$",
    'copt-aux1': r"$c_{opt} (S=1)$",
    'copt-aux10': r"$c_{opt} (S=10)$",
    'copt-aux50': r"$c_{opt} (S=50)$",
    'copt-aux100': r"$c_{opt} (S=100)$",
    'reinforce': r"$\operatorname{REINFORCE}$",
    'vimco-arithmetic': r"$\operatorname{VIMCO}_{\operatorname{arithmetic}}}$",
    'vimco-geometric': r"$\operatorname{VIMCO}_{\operatorname{geometric}}}$",
    'tvo': r"$\operatorname{TVO}$",
    'ww': r"$\operatorname{RWS}$",
    'pathwise': r"$\operatorname{PATHWISE}$"
}

LINE_STYLES = 10 * ["-", "--", ":", "-."]
MARKERS = 10 * ["o", "v", "^", "s", "P", "X", "D", "+", "x"]
DASH_STYLES = 10 * ["",
                    (4, 1.5),
                    (1, 1),
                    (3, 1, 1.5, 1),
                    (5, 1, 1, 1),
                    (5, 1, 2, 1, 2, 1),
                    (2, 2, 3, 1.5),
                    (1, 2.5, 3, 1.2)]

LOG_PLOT_RULES = {
    'iw': 'log',
    'grads/snr': 'log',
    'grads/dsnr': 'log',
    'grads/variance': 'log',
    'grads/magnitude': 'log',
    'gmm/posterior_mse': 'log',
    'gmm/prior_mse': 'log',
    'gaussian_toy/mse_A': 'log',
    'gaussian_toy/mse_b': 'log',
    'gaussian_toy/mse_mu': 'log'
}

METRIC_DISPLAY_NAME = {
    'iw': r"$K$",
    'c_iw': r"$K$",
    'loss/L_k': r"$\log p_{\theta}(x)$",
    'loss/elbo': r"$\operatorname{ELBO}$",
    'loss/kl': r"$\operatorname{KL}(q_{\phi}(z | x) || p(z))$",
    'loss/nll': r"$- \log p_{\theta}(x | z)$",
    'loss/r_ess': r"$\operatorname{ESS} / K$",
    'loss/ess': r"$\operatorname{ESS}$",
    'loss/kl_q_p': r"$\mathrm{KL}\left(Q || P\right)$",
    'grads/variance': r"$\operatorname{Var}(\Delta(\phi))$",
    'grads/snr': r"$\operatorname{SNR}(\Delta(\phi))$",
    'grads/dsnr': r"$\operatorname{DSNR}(\Delta(\phi))$",
    'grads/magnitude': r"$ | E[\Delta(\phi)] | $",
    'grads/direction': r"$\operatorname{cosine}( \Delta(\phi) ,  \Delta^{oracle}(\phi) )$",
    'gmm/posterior_mse': r"$\Vert q_{\phi}(z | x) - p_{\theta_{true}}(z | x) \Vert_2 $",
    'gmm/prior_mse': r"$\Vert p_{\theta}(z) - p_{\theta_{true}}(z) \Vert_2 $",
    'gaussian_toy/mse_A': r"$\Vert A - A^*  \Vert_2$",
    'gaussian_toy/mse_b': r"$\Vert b - b^*  \Vert_2$",
    'gaussian_toy/mse_mu': r"$\Vert \mu - \mu^*  \Vert_2$",

}