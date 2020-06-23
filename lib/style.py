import seaborn as sns
import pprint
import matplotlib
pp = pprint.PrettyPrinter(indent=4)

def set_style():
    # matplotlib.rc('font', family='serif', serif='Bookman')
    matplotlib.rc('text', usetex=True)
    #matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    matplotlib.rcParams['text.latex.preamble'] = [
        r"\usepackage{amsmath}"
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ]

    sns.set_style("darkgrid",  {"axes.facecolor": ".96", "xtick.bottom": True, "ytick.left":True, "xtick.color": "0.3", "ytick.color": "0.3"})
    # sns.set(style="ticks")
    sns.set_context("paper", font_scale=1.6, rc={"lines.linewidth": 1.2, "lines.markersize": 12, 'text.latex.preamble': r"\usepackage{amsmath}"})

    print(100*"==")
    print("Axes style")
    print(100 * "==")
    pp.pprint(sns.axes_style())
    print(100 * "==")

DPI = 200
PLOT_WIDTH = 5
PLOT_TOTAL_WIDTH = None
PLOT_HEIGHT = 3#2.5
STEP_FORMAT = '{x:.0e}'

ESTIMATOR_STYLE = {
    'copt-alpha0': {'color': "#E6C445", 'marker': ">", 'linestyle': "-"},
    'copt-alpha1': {'color': "#B88A25", 'marker': "<", 'linestyle': "-"},
    'copt-aux1': {'color': "#E89C66", 'marker': "^", 'linestyle': ":"},
    'copt-aux10': {'color': "#D15C2A", 'marker': "^", 'linestyle': "-"},
    'copt-aux50': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'copt-aux100': {'color': "#954026", 'marker': "^", 'linestyle': "-"},
    'reinforce': {'color': "#7A9396", 'marker': "o", 'linestyle': "-"},
    'vimco-arithmetic': {'color': "#6CAD8E", 'marker': "h", 'linestyle': "-"},
    'vimco-geometric': {'color': "#5A9177", 'marker': "H", 'linestyle': "-"},
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
    'vimco',
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
    'copt-alpha0': r"$\operatorname{OVIS}_{\sim}\ (\gamma = 1)$", # carreful here 0 -> 1
    'copt-alpha1': r"$\operatorname{OVIS}_{\boldsymbol{\sim}}\ (\gamma = 0)$",
    'copt-aux1': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=1)$",
    'copt-aux10': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=10)$",
    'copt-aux50': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=50)$",
    'copt-aux100': r"$\operatorname{OVIS}_{\operatorname{MC}}\ (S=100)$",
    'reinforce': r"$\operatorname{REINFORCE}$",
    'vimco': r"$\operatorname{VIMCO}$",
    'vimco-arithmetic': r"$\operatorname{VIMCO}$", #r"$\operatorname{VIMCO}_{\operatorname{arithmetic}}$",
    'vimco-geometric': r"$\operatorname{VIMCO}_{\operatorname{geometric}}$",
    'tvo': r"$\operatorname{TVO}$",
    'ww': r"$\operatorname{RWS}$",
    'pathwise': r"$\operatorname{pathwise}\ (\operatorname{IWAE})$"
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
    'gamma' : 'log',
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
    'active_units/au': r"active units"

}


def format_estimator_name(name):
    if name == 'copt':
        return 'copt-alpha1'
    if name == 'copt-alpha0':
        return 'copt-alpha0'
    elif name == 'copt-vimco':
        return 'copt-alpha1'
    elif name == 'vimco':
        return 'vimco-arithmetic'
    elif name == 'tvo-config1' or name == 'tvo-config2':
        return 'tvo'
    elif 'tvo' in name:
        return 'tvo'
    elif name == 'pathwise-iwae':
        return 'pathwise'
    else:
        return name