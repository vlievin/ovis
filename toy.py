import argparse
import json
import os
import socket
import traceback
from shutil import rmtree

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lib import ToyVAE
from lib.config import get_config
from lib.estimators import VariationalInference
from lib.gradients import get_gradients_statistics
from lib.logging import get_loggers
from lib.plotting import markers
from lib.utils import notqdm

_sep = os.get_terminal_size().columns * "-"

parser = argparse.ArgumentParser()

default_iws = np.geomspace(start=3, stop=10000, num=12)[::-1]
default_iws = ",".join([str(int(k)) for k in default_iws])

print("# default_iws:", default_iws)

# run directory, id and seed
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--data_root', default='data/', help='directory to store the data')
parser.add_argument('--exp', default='gaussian-toy-variance-0.1', help='experiment directory')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--seed', default=13, type=int, help='random seed')
parser.add_argument('--workers', default=1, type=int, help='dataloader workers')
parser.add_argument('--rm', action='store_true', help='delete previous run')
parser.add_argument('--silent', action='store_true', help='silence tqdm')
parser.add_argument('--deterministic', action='store_true', help='use deterministic backend')
parser.add_argument('--sequential_computation', action='store_true',
                    help='compute each iw sample sequential during validation')

# estimator
parser.add_argument('--estimators',
                    default='copt-arithmetic,vimco-arithmetic,pathwise',
                    help='[vi, reinforce, vimco, gs, st-gs]')
parser.add_argument('--iws', default=default_iws,
                    help='number of Importance-Weighted samples')
parser.add_argument('--iw_valid', default=1000, type=int, help='number of iw samples for testing')

# noise perturbation for the parameters
parser.add_argument('--noise', default=0.01, type=float, help='scale of the noise added to the optimal parameters')

# evaluation of the gradients
parser.add_argument('--key_filter', default='tensor:b', type=str,
                    help='identifiant of the parameters/tensor for the gradients analysis')
parser.add_argument('--mc_samples', default=100, type=int, help='number of samples for gradients evaluation')
parser.add_argument('--max_points', default=0, type=int, help='number of data points to evaluate the grads in')
parser.add_argument('--batch_size', default=80000, type=int,
                    help='number of samples per batch size used during evaluation')
parser.add_argument('--grads_dist', action='store_true', help='plot distributions of gradients for each parameter')
parser.add_argument('--max_params', default=3, type=int, help='max number of parameters to analyze')

# dataset
parser.add_argument('--npoints', default=1024, type=int, help='number of datapoints')
parser.add_argument('--D', default=20, type=int, help='number of latent variables')

opt = parser.parse_args()

if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if opt.silent:
    tqdm = notqdm

# defining the run identifier
run_id = f"toy-seed{opt.seed}-noise{opt.noise}-mc{opt.mc_samples}-key{opt.key_filter}-pts{opt.npoints}-D{opt.D}"
if opt.id != "":
    run_id += f"-{opt.id}"
_exp_id = f"toy-{opt.exp}-{opt.seed}"

# defining the run directory
logdir = os.path.join(opt.root, opt.exp)
logdir = os.path.join(logdir, run_id)
if os.path.exists(logdir):
    if opt.rm:
        rmtree(logdir)
        os.makedirs(logdir)
else:
    os.makedirs(logdir)

# save configuration
with open(os.path.join(logdir, 'config.json'), 'w') as fp:
    _opt = vars(opt)
    _opt['hostname'] = socket.gethostname()
    fp.write(json.dumps(_opt, default=lambda x: str(x), indent=4))

try:
    # define logger
    base_logger, train_logger, valid_logger, test_logger = get_loggers(logdir)
    base_logger.info(f"Torch version: {torch.__version__}")

    # setting the random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # define model
    torch.manual_seed(opt.seed)
    model = ToyVAE((opt.D,), None, None, None)

    # valid estimator (it is important that all models are evaluated using the same evaluator)
    config_valid = {'tau': 0, 'zgrads': False}
    estimator_valid = VariationalInference(mc=1, iw=opt.iw_valid, sequential_computation=opt.sequential_computation)

    # get device and move models
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    model.to(device)
    estimator_valid.to(device)

    # parse estimators
    estimators = opt.estimators.replace(" ", "").split(",")
    iws = list(sorted([eval(k) for k in opt.iws.replace(" ", "").split(",")], reverse=True))

    # generate the dataset
    # model.mu.data = torch.randn_like(model.mu.data)
    # x = model.sample_from_prior(N=opt.npoints)['px'].sample()

    from lib.datasets.gaussian_toy import GaussianToyDataset

    dset = GaussianToyDataset()
    dset.data = dset.data.to(device)
    x = dset.data

    # evaluate model
    torch.manual_seed(opt.seed)
    loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    base_logger.info(
        f"Before init. | L_{estimator_valid.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, KL = {diagnostics['loss']['kl'].mean().item():.6f}, NLL = {diagnostics['loss']['nll'].mean().item():.6f}")

    # initizalize model using the optimal parameters
    mu = x.mean(dim=0, keepdim=True).data
    model.mu.data = mu.data.view_as(model.mu.data)  # mu^*
    model.A.data = 0.5 * torch.eye(x.shape[1], device=x.device).view_as(model.A.data)  # A = I / 2
    model.b.data = 0.5 * mu.view_as(model.b.data)  # b = mu^* / 2

    true_grads = x.mean(dim=0, keepdim=True).data - model.b.data if opt.key_filter == 'tensor:b' else None
    _, grads_idx = true_grads[0].abs().sort(descending=True)

    # evaluate model
    torch.manual_seed(opt.seed)
    loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    base_logger.info(
        f"After init. | L_{estimator_valid.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, KL = {diagnostics['loss']['kl'].mean().item():.6f}, NLL = {diagnostics['loss']['nll'].mean().item():.6f}")

    # add perturbation to the weights
    model.mu.data = model.mu.data + opt.noise * torch.randn_like(model.mu.data)
    model.A.data = model.A.data + opt.noise * torch.randn_like(model.A.data)
    model.b.data = model.b.data + opt.noise * torch.randn_like(model.b.data)

    # evaluate model
    torch.manual_seed(opt.seed)
    loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    base_logger.info(
        f"After perturbation | L_{estimator_valid.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, KL = {diagnostics['loss']['kl'].mean().item():.6f}, NLL = {diagnostics['loss']['nll'].mean().item():.6f}")

    # gradients analysis args and config
    meta = {'seed': opt.seed, 'noise': opt.noise, 'mc_samples': opt.mc_samples,
            **{k: v.mean().item() for k, v in diagnostics['loss'].items()}}
    grad_args = {'seed': opt.seed, 'batch_size': opt.batch_size, 'n_samples': opt.mc_samples,
                 'key_filter': opt.key_filter, 'true_grads': true_grads}

    x_eval = x if opt.max_points < 1 else x[:opt.max_points]
    data = []
    grads = []
    for estimator_id in estimators:
        for iw in tqdm(iws, desc=f"{estimator_id} : iws"):
            # create estimator
            Estimator, config = get_config(estimator_id)
            estimator = Estimator(baseline=None, mc=1, iw=iw)
            estimator.to(device)

            # setting the random seed
            torch.manual_seed(opt.seed)

            # evalute variance of the gradients
            analysis_data, grads_ = get_gradients_statistics(estimator, model, x_eval, return_grads=opt.grads_dist, **grad_args, **config)

            # log grads info
            grad_data = analysis_data.get('grads', {})
            base_logger.info(
                f"{estimator_id}, iw = {iw} | snr = {grad_data.get('snr', 0.):.3E}, variance = {grad_data.get('variance', 0.):.3E}, magnitude = {grad_data.get('magnitude', 0.):.3E}, dir = {grad_data.get('direction', 0.):.3E}")
            snr_data = analysis_data.get('snr', {})
            base_logger.info(
                f"{estimator_id}, iw = {iw} | snr | p5 = {snr_data.get('p5', 0.):.3E}, p25 = {snr_data.get('p25', 0.):.3E}, p50 = {snr_data.get('p50', 0.):.3E}, p75 = {snr_data.get('p75', 0.):.3E}, p95 = {snr_data.get('p95', 0.):.3E}")

            # stor results
            data += [{
                'estimator': estimator_id,
                'iw': iw,
                **{f"grads-{k}": v.item() for k, v in grad_data.items()},
                **{f"snr-{k}": v.item() for k, v in snr_data.items()},
                **meta
            }]

            # store grads
            if opt.grads_dist:
                grads_ = grads_.view(-1, grads_.shape[-1]).transpose(1, 0)
                grads_ = grads_[grads_idx] # reindex by `true_grads` magnitude
                for p, grads_p in enumerate(grads_): # for each parameter
                    if p < opt.max_params:
                        for g in grads_p:
                            grads += [{'param': p, 'estimator': estimator_id, 'iw': iw, 'grad': g.item()}]


    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(logdir, 'data.csv'))
    base_logger.info(f"# path = {os.path.abspath(logdir)}")

    if len(grads):
        grads = pd.DataFrame(grads)

    # plotting
    param_name = {'tensor:b': "b", 'tensor:qlogits': "\phi"}.get(opt.key_filter, "\theta")
    metrics = ['snr-p50', 'grads-variance', 'grads-magnitude', 'grads-direction']
    metrics_formaters = [lambda p: f"$SNR_K({param_name}) $",
                         lambda p: f"$Var \Delta_K({param_name}) $",
                         lambda p: f"$| \Delta_K({param_name}) |$",
                         lambda p: f"$cosine( \Delta_K({param_name}), \Delta_K({param_name})^* )$",
                         ]
    nrows = 1
    ncols = len(metrics)
    hue_order = list(df['estimator'].unique())
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols, 3 * nrows))
    for k, metric in enumerate(metrics):
        ax = axes[k]
        # ax.set(yscale="log")
        # ax.set(xscale="log", yscale="log")
        # sns.pointplot(x="iw", y=metric, hue="estimator", data=df, ax=ax, hue_order=hue_order, capsize=.2)

        # legend = ax.legend()
        #         # if k < len(metrics) - 1 and legend is not None:
        #         #     legend.remove()

        if k == 0:
            iws = list(sorted(df['iw'].unique()))
            expected_max = [1e1 / k**0.5 for k in iws]
            expected_min = [1e-1 / k**0.5 for k in iws]
            ax.loglog(iws, expected_min, ":", color="gray", basex=10, basey=10)
            ax.loglog(iws, expected_max, ":", color="gray", basex=10, basey=10)

        for e, estimator_id in enumerate(estimators):
            sub_df = df[df['estimator'] == estimator_id]
            iws = sub_df['iw'].values
            values = sub_df[metric].values

            if "direction" in metric:
                ax.plot(iws, values, linestyle="-", marker=markers[e], markersize=10, label=estimator_id)
                ax.set_xscale('log')
            else:
                ax.loglog(iws, values, linestyle="-", marker=markers[e], markersize=10, label=estimator_id, basex=10,
                          basey=10)

        ax.set_xlabel("$K$")
        ax.set_ylabel(metrics_formaters[k](param_name))

        if k == len(metrics) - 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "gradients.png"))
    plt.show(block=False)
    # plt.pause(10)
    plt.close()


    # gradients dist
    if len(grads):
        print("GRADS:", len(grads))
        def agg(s):
            return f"{np.mean(s):.3f} +/- {np.std(s):.3f} (n={len(s)})"
        print(grads.pivot_table(columns="estimator", index=["param","iw"], values="grad", aggfunc=agg))

        # pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(grads, row="estimator", col="param", hue="iw", aspect=1.5, height=2)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "grad", clip_on=False, shade=True, alpha=0.2)
        #g.map(sns.kdeplot, "grad", clip_on=False, color="w", lw=2, bw=.2)
        # g.map(plt.axhline, y=0, lw=2, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, f"{x} : {label}", color=color,
                    ha="left", va="center", transform=ax.transAxes)
        # g.map(label, "grad")
        #
        # # Set the subplots to overlap
        # g.fig.subplots_adjust(hspace=-.25)
        #
        # # Remove axes details that don't play well with overlap
        # g.set_titles("")
        g.set(yticks=[])
        #g.set_ylabels("iw")
        # g.map(plt.set_ylabels, "iw")
        g.despine(bottom=True, left=True)

        a, b = np.percentile(grads["grad"].values, [25, 75])
        plt.xlim([a - 2.5 * (b-a), b + 2.5 * (b-a)])

        plt.legend()
        plt.tight_layout()

        plt.show()


except KeyboardInterrupt:
    print("## KEYBOARD INTERRUPT")
    with open(os.path.join(logdir, "success.txt"), 'w') as f:
        f.write(f"Failed. Interrupted (keyboard).")

except Exception as ex:
    print("## FAILED. Exception:")
    print("--------------------------------------------------------------------------------")
    traceback.print_exception(type(ex), ex, ex.__traceback__)
    print("--------------------------------------------------------------------------------")
    print("\nException: ", ex, "\n")
    with open(os.path.join(logdir, "success.txt"), 'w') as f:
        f.write(f"Failed. Exception : \n{ex}")
