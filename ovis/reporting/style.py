import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.utils import Header

DPI = 200
PLOT_WIDTH = 5
PLOT_TOTAL_WIDTH = None
PLOT_HEIGHT = 3  # 2.5
STEP_FORMAT = '{x:.0e}'


def set_matplotlib_style():
    """
    Set a custom `matplotlib` style and enable `Latex`.
    """

    try:
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble'] = [
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r'\usepackage{siunitx}',
            r'\sisetup{detect-all}',
            r'\usepackage{helvet}',
            r'\usepackage{sansmath}',
            r'\sansmath'
        ]

        # test Latex
        plt.scatter([0, 1], [0, 1])
        plt.xlabel(r"$K$")
        plt.ylabel(r"$\mathcal{L}$")
        plt.savefig('.test_latex.png')
        plt.close()

    except Exception as ex:
        matplotlib.rc('text', usetex=False)
        with Header("@ style.py: couldn't load Latex"):
            print(ex)

    sns.set_style("darkgrid", {"axes.facecolor": ".96", "xtick.bottom": True, "ytick.left": True, "xtick.color": "0.3",
                               "ytick.color": "0.3"})

    sns.set_context("paper", font_scale=1.6,
                    rc={"lines.linewidth": 1.2, "lines.markersize": 12, 'text.latex.preamble': r"\usepackage{amsmath}"})


DATASET_DISPLAY_NAME = {
    'binmnist': 'Binarized MNIST',
    'fashion': 'Binarized F-MNIST',
    'omniglot': 'Omniglot',
}

ESTIMATOR_STYLE = {
    'ovis-gamma1': {'color': "#E6C445", 'marker': ">", 'linestyle': "-"},
    'ovis-arithmetic-gamma1': {'color': "#B07BF9", 'marker': 4, 'linestyle': "-"},
    'ovis-arithmetic-gamma0': {'color': "#F97B9B", 'marker': 8, 'linestyle': "-"},
    'ovis-gamma0': {'color': "#B88A25", 'marker': "<", 'linestyle': "-"},
    'ovis-S1': {'color': "#E89C66", 'marker': "^", 'linestyle': ":"},
    'ovis-S10': {'color': "#D15C2A", 'marker': "^", 'linestyle': "-"},
    'ovis-S50': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'ovis-S100': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'ovis-S0.1': {'color': "#F2CBBB", 'marker': "^", 'linestyle': ":"},
    'ovis-S0.2': {'color': "#E39878", 'marker': "^", 'linestyle': ":"},
    'ovis-S0.3': {'color': "#DA7144", 'marker': "^", 'linestyle': ":"},
    'ovis-S0.5': {'color': "#99431E", 'marker': "^", 'linestyle': ":"},
    'ovis-S0.7': {'color': "#512315", 'marker': "^", 'linestyle': ":"},
    'ovis': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'reinforce': {'color': "#7A9396", 'marker': "o", 'linestyle': "-"},
    'vimco-arithmetic': {'color': "#6CAD8E", 'marker': "h", 'linestyle': "-"},
    'vimco-geometric': {'color': "#5A9177", 'marker': "H", 'linestyle': "-"},
    'tvo': {'color': "#336DA2", 'marker': "P", 'linestyle': "-"},
    'wake-wake': {'color': "#77B6D7", 'marker': "X", 'linestyle': "-"},
    'wake-sleep': {'color': "#7091d4", 'marker': "x", 'linestyle': "-"},
    'pathwise-iwae': {'color': "#968FA1", 'marker': "s", 'linestyle': "-"},
    'pathwise-vae': {'color': "#554c5e", 'marker': "d", 'linestyle': "-"},
    'stl': {'color': "#c075d9", 'marker': 2, 'linestyle': "-"},
    'dreg': {'color': "#bd2f82", 'marker': 3, 'linestyle': "-"},
    'gs': {'color': "#968FA1", 'marker': "*", 'linestyle': "-"}
}

ESTIMATOR_ORDER = [
    'ovis-gamma0',
    'ovis-gamma1',
    'ovis-arithmetic-gamma1',
    'ovis-arithmetic-gamma0',
    'ovis',
    'ovis-S100',
    'ovis-S50',
    'ovis-S10',
    'ovis-S1',
    'ovis-S0.1',
    'ovis-S0.1',
    'ovis-S0.2',
    'ovis-S0.3',
    'ovis-S0.5',
    'ovis-S0.7',
    'tvo',
    'wake-wake',
    'wake-sleep',
    'vimco',
    'vimco-arithmetic',
    'vimco-geometric',
    'reinforce',
    'pathwise-iwae',
    'stl',
    'dreg',
    'pathwise-vae',
    'gs'
]

ESTIMATOR_GROUPS = {
    'ovis-gamma0': 0,
    'ovis-gamma1': 0,
    'ovis-arithmetic-gamma0': 0,
    'ovis-arithmetic-gamma1': 0,
    'ovis': 1,
    'ovis-S100': 1,
    'ovis-S50': 1,
    'ovis-S10': 1,
    'ovis-S1': 1,
    'ovis-S0.1': 1,
    'ovis-S0.2': 1,
    'ovis-S0.3': 1,
    'ovis-S0.5': 1,
    'ovis-S0.7': 1,
    'wake-wake': 2,
    'wake-sleep': 2,
    'tvo': 2,
    'vimco-arithmetic': 3,
    'vimco-geometric': 3,
    'reinforce': 3,
    'pathwise-iwae': 4,
    'stl': 4,
    'dreg': 4,
    'pathwise-vae': 4,
    'gs': 4
}

ESTIMATOR_DISPLAY_NAME = {
    'ovis-gamma1': r"$\operatorname{OVIS}_{\sim}\ (\gamma = 1)$",
    'ovis-arithmetic-gamma1': r"$\operatorname{OVIS}_{\sim}\ (\gamma = 1) \operatorname{no\ clip.}$",
    'ovis-arithmetic-gamma0': r"$\operatorname{OVIS}_{\sim}\ (\gamma = 0) \operatorname{no\ clip.}$",
    'ovis-gamma0': r"$\operatorname{OVIS}_{\sim}\ (\gamma = 0)$",
    'ovis-S1': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=1)$",
    'ovis-S10': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=10)$",
    'ovis-S50': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=50)$",
    'ovis-S0.1': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=0.1K')$",
    'ovis-S0.2': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=0.2K')$",
    'ovis-S0.3': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=0.3K')$",
    'ovis-S0.5': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=0.5K')$",
    'ovis-S0.7': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=0.7K')$",
    'ovis': r"$\operatorname{OVIS}_{\operatorname{MC}}$",
    'ovis-S100': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=100)$",
    'reinforce': r"$\operatorname{REINFORCE}$",
    'vimco': r"$\operatorname{VIMCO}$",
    'vimco-arithmetic': r"$\operatorname{VIMCO} \operatorname{arith.}$",  # r"$\operatorname{VIMCO}$",
    'vimco-geometric': r"$\operatorname{VIMCO} \operatorname{geo.}$",
    'tvo': r"$\operatorname{TVO}$",
    'wake-wake': r"$\operatorname{RWS}$",
    'wake-sleep': r"$\operatorname{RWS}(\operatorname{WS})$",
    'pathwise-iwae': r"$\operatorname{pathwise}\ (\operatorname{IWAE})$",
    'stl': r"$\operatorname{pathwise}\ (\operatorname{IWAE-STL})$",
    'dreg': r"$\operatorname{pathwise}\ (\operatorname{IWAE-DReG})$",
    'pathwise-vae': r"$\operatorname{pathwise}\ (\operatorname{VAE})$",
    'gs': r"$\operatorname{Concrete})$"
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
    'gamma': 'log',
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
    'gamma': r"1- $\alpha$",
    'gamma_min': r"$1 - \alpha_{\operatorname{min}}$",
    'c_iw': r"$K$",
    'loss/L_k': r"$\log \hat{p}_{\theta}(\mathbf{x})$",
    'loss/elbo': r"$\operatorname{ELBO}$",
    'loss/kl': r"$\mathrm{KL} \left( q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z}) \right)$",
    'loss/nll': r"$- \log p_{\theta}(\mathbf{x} | \mathbf{z})$",
    'loss/r_ess': r"$\operatorname{ESS} / K$",
    'loss/ess': r"$\operatorname{ESS}$",
    'loss/kl_q_p': r"$\mathrm{KL} \left(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z} | \mathbf{x})\right)$",
    'grads/variance': r"$\frac{1}{M} \sum_i \operatorname{Var}\left[ g_i \right]$",
    'grads/snr': r"$\frac{1}{M} \sum_i \operatorname{SNR}_i$",
    'grads/dsnr': r"$\frac{1}{M} \sum_i \operatorname{DSNR}_i$",
    'grads/magnitude': r"$ \frac{1}{M} \sum_i | E\left[ g_i \right] | $",
    'grads/direction': r"$\operatorname{cosine}( \Delta(\phi) ,  \Delta^{oracle}(\phi) )$",
    'gmm/posterior_mse': r"$\Vert q_{\phi}( z | x ) - p_{\theta^\star}(z|x) \Vert_2 $",
    'gmm/prior_mse': r"$\Vert p_{\theta}(z) - p_{\theta^\star}(z) \Vert_2 $",
    'gaussian_toy/mse_A': r"$\Vert A - A^\star  \Vert_2$",
    'gaussian_toy/mse_b': r"$\Vert \mathbf{b} - \mathbf{b}^\star  \Vert_2$",
    'gaussian_toy/mse_mu': r"$\Vert \mu - \mu^*\star \Vert_2$",
    'gaussian_toy/mse_phi': r"$\Vert \phi - \phi^\star  \Vert_2$",
    'active_units/au': r"active units",
    'max_memory': r'max. memory (MB)',
    'elapsed_time': r'epoch time (s.)'

}


def format_estimator_name(name):
    """add special rules to format the estimator name"""
    name = name.replace("-exclusive", "")

    if 'tvo' in name:
        return 'tvo'
    else:
        return name
