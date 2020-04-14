import argparse
import json
import os
import socket
import traceback
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from lib import ToyVAE
from lib.config import get_config
from lib.estimators import VariationalInference
from lib.gradients import get_gradients_statistics
from lib.logging import get_loggers
from lib.plotting import markers, PLOT_WIDTH, PLOT_HEIGHT
from lib.utils import notqdm

colors = sns.color_palette()
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
parser.add_argument('--noise', default=0.05, type=str, help='scale of the noise added to the optimal parameters')

# evaluation of the gradients
parser.add_argument('--key_filter', default='tensor:b', type=str,
                    help='identifiant of the parameters/tensor for the gradients analysis')
parser.add_argument('--mc_samples', default=100, type=int, help='number of samples for gradients evaluation')
parser.add_argument('--max_points', default=0, type=int, help='number of data points to evaluate the grads in')
parser.add_argument('--batch_size', default=80000, type=int,
                    help='number of samples per batch size used during evaluation')
parser.add_argument('--grads_dist', action='store_true', help='plot distributions of gradients for each parameter')
parser.add_argument('--use_all_params', action='store_true', help='look a the aggregated dist of grads')

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

    if opt.max_points < 1:
        x_eval = x
    elif opt.max_points == 1:
        x_eval = x[:opt.max_points] # x.mean(dim=0, keepdim=True)
    else:
        x_eval = x[:opt.max_points]
    data = []
    grads = []
    noises = [eval(x) for x in opt.noise.split(",")]
    for noise in noises:

        # initizalize model using the optimal parameters
        mu = x.mean(dim=0, keepdim=True).data
        model.mu.data = mu.data.view_as(model.mu.data)  # mu^*
        model.A.data = 0.5 * torch.eye(x.shape[1], device=x.device).view_as(model.A.data)  # A = I / 2
        model.b.data = 0.5 * mu.view_as(model.b.data)  # b = mu^* / 2

        true_grads = x.mean(dim=0, keepdim=True).data - model.b.data if opt.key_filter == 'tensor:b' else None
        # _, grads_idx = true_grads[0].abs().sort(descending=True)
        grads_idx = None

        # evaluate model
        torch.manual_seed(opt.seed)
        loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
        base_logger.info(
            f"After init. | L_{estimator_valid.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, KL = {diagnostics['loss']['kl'].mean().item():.6f}, NLL = {diagnostics['loss']['nll'].mean().item():.6f}")

        # add perturbation to the weights
        model.mu.data = model.mu.data + noise * torch.randn_like(model.mu.data)
        model.A.data = model.A.data + noise * torch.randn_like(model.A.data)
        model.b.data = model.b.data + noise * torch.randn_like(model.b.data)

        # evaluate model
        torch.manual_seed(opt.seed)
        loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
        base_logger.info(
            f"After perturbation | L_{estimator_valid.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, KL = {diagnostics['loss']['kl'].mean().item():.6f}, NLL = {diagnostics['loss']['nll'].mean().item():.6f}")

        # gradients analysis args and config
        meta = {'seed': opt.seed, 'noise': noise, 'mc_samples': opt.mc_samples,
                **{k: v.mean().item() for k, v in diagnostics['loss'].items()}}
        grad_args = {'seed': opt.seed, 'batch_size': opt.batch_size, 'n_samples': opt.mc_samples,
                     'key_filter': opt.key_filter, 'true_grads': true_grads}

        for estimator_id in estimators:
            for iw in tqdm(iws, desc=f"{estimator_id} : iws"):

                # create estimator
                if "-mc" in estimator_id:
                    Estimator, config = get_config(estimator_id.replace("-mc", ""))
                    _mc, _iw = iw, 1
                else:
                    Estimator, config = get_config(estimator_id)
                    _mc, _iw = 1, iw


                estimator = Estimator(baseline=None, mc=_mc, iw=_iw)
                estimator.to(device)

                # setting the random seed
                torch.manual_seed(opt.seed)

                # evalute variance of the gradients
                analysis_data, grads_ = get_gradients_statistics(estimator, model, x_eval, return_grads=opt.grads_dist,
                                                                 use_dsnr=True, **grad_args, **config)

                # log grads info
                grad_data = analysis_data.get('grads', {})
                base_logger.info(
                    f"{estimator_id}, iw = {iw} | snr = {grad_data.get('snr', 0.):.3E}, dsnr = {grad_data.get('dsnr', 0.):.3E}, variance = {grad_data.get('variance', 0.):.3E}, magnitude = {grad_data.get('magnitude', 0.):.3E}, dir = {grad_data.get('direction', 0.):.3E}")
                snr_data = analysis_data.get('snr', {})
                base_logger.info(
                    f"{estimator_id}, iw = {iw} | snr | p5 = {snr_data.get('p5', 0.):.3E}, p25 = {snr_data.get('p25', 0.):.3E}, p50 = {snr_data.get('p50', 0.):.3E}, p75 = {snr_data.get('p75', 0.):.3E}, p95 = {snr_data.get('p95', 0.):.3E}")

                # stor results
                data += [{
                    'estimator': estimator_id,
                    'iw': iw,
                    **{f"grads-{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in grad_data.items()},
                    **{f"snr-{k}": v.item() for k, v in snr_data.items()},
                    **meta
                }]

                # store grads
                if opt.grads_dist:
                    grads_ = grads_.view(-1, grads_.shape[-1]).transpose(1, 0)

                    # take the grads with the maximum expected value
                    if grads_idx is None:
                        _, grads_idx = grads_.mean(1).abs().sort(descending=True)

                    grads_ = grads_[grads_idx]  # reindex by `true_grads` magnitude

                    # get only positive biases
                    u = grads_.mean(1, keepdim=True)
                    dir = 2 * (u > 0).float() - 1.
                    grads_ = grads_ * dir

                    if opt.use_all_params:
                        # return gradients for all parameters
                        for g in grads_.view(-1):
                            grads += [{'noise': noise, 'param': 'all', 'estimator': estimator_id, 'iw': iw, 'grad': g.item()}]
                    else:
                        # return gradients for the first param
                        for g in grads_[0, :]:
                            grads += [{'noise': noise, 'param': 'all', 'estimator': estimator_id, 'iw': iw, 'grad': g.item()}]

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(logdir, 'data.csv'))
    base_logger.info(f"# path = {os.path.abspath(logdir)}")

    if len(grads):
        grads = pd.DataFrame(grads)

    # plotting
    param_name = {'tensor:b': "b", 'tensor:qlogits': "\phi"}.get(opt.key_filter, "\theta")
    metrics = ['grads-snr', 'grads-dsnr', 'grads-variance', 'grads-magnitude'] #, 'grads-direction'
    metrics_formaters = [lambda p: f"$SNR_K({param_name}) $",
                         lambda p: f"$DSNR_K({param_name}) $",
                         lambda p: f"$Var \Delta_K({param_name}) $",
                         lambda p: f"$| \Delta_K({param_name}) |$"
                         # lambda p: f"$cosine( \Delta_K({param_name}), \Delta_K({param_name})^* )$",
                         ]
    nrows = len(noises)
    ncols = len(metrics)
    hue_order = list(df['estimator'].unique())
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows), sharex='col', sharey='col', squeeze=False)
    for n, noise in enumerate(sorted(noises)):

        noise_df = df[df['noise'] == noise]
        for k, metric in enumerate(metrics):
            ax = axes[n,k]
            # ax.set(yscale="log")
            # ax.set(xscale="log", yscale="log")
            # sns.pointplot(x="iw", y=metric, hue="estimator", data=df, ax=ax, hue_order=hue_order, capsize=.2)

            # legend = ax.legend()
            #         # if k < len(metrics) - 1 and legend is not None:
            #         #     legend.remove()

            if k == 0:
                iws = list(sorted(df['iw'].unique()))
                expected_max = [1e1 / k ** 0.5 for k in iws]
                expected_min = [1e-1 / k ** 0.5 for k in iws]
                ax.loglog(iws, expected_min, ":", color="gray", basex=10, basey=10)
                ax.loglog(iws, expected_max, ":", color="gray", basex=10, basey=10)

            for e, estimator_id in enumerate(estimators):
                sub_df = noise_df[noise_df['estimator'] == estimator_id]
                iws = sub_df['iw'].values
                values = sub_df[metric].values

                if "direction" in metric:
                    ax.plot(iws, values, linestyle="-", marker=markers[e], markersize=10, label=estimator_id)
                    ax.set_xscale('log')
                else:
                    ax.loglog(iws, values, linestyle="-", marker=markers[e], markersize=10, label=estimator_id, basex=10,
                              basey=10)

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
    plt.show(block=False)
    # plt.pause(10)
    plt.close()

    # gradients dist
    if len(grads):
        print("GRADS:", len(grads))


        def agg(s):
            return f"{np.mean(s):.3f} +/- {np.std(s):.3f} (n={len(s)})"


        print(grads.pivot_table(columns="estimator", index=["noise", "iw"], values="grad", aggfunc=agg))
        print("------------------------------------------------------------------------")

        noises = grads['noise'].unique()
        estimators = grads['estimator'].unique()
        iws = list(sorted(grads["iw"].unique()))

        ncols = len(noises)
        nrows = len(iws)

        if ncols > 1:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(PLOT_WIDTH * ncols, PLOT_HEIGHT * nrows),
                                     sharex='col')
        else:
            fig, axes = plt.subplots(nrows=1, ncols=nrows, figsize=(PLOT_WIDTH * nrows, PLOT_HEIGHT * 1), sharex='row', sharey='row')

        for j, noise in enumerate(noises):
            grads_p = grads[grads["noise"] == noise]

            # filter data
            values = grads_p["grad"].values
            k_out = 3.0
            a, b = np.percentile(values, [25, 75])
            y_b = b + k_out * (b - a)
            y_a = a - k_out * (b - a)
            filtered_grads_p = grads_p[(grads_p["grad"] <= y_b) & (grads_p["grad"] >= y_a)]

            for i, iw in enumerate(iws):

                grads_p_iw = grads_p[grads_p["iw"] == iw]
                filtered_grads_p_iw = filtered_grads_p[filtered_grads_p["iw"] == iw]

                ax = axes[i, j] if ncols > 1 else axes[i]
                ax.axvline(x=0, color="gray", alpha=0.9, linestyle="-")

                for e, estimator_id in enumerate(estimators):
                    color = colors[e]

                    filtered_data = filtered_grads_p_iw[filtered_grads_p_iw["estimator"] == estimator_id]['grad'].values
                    sns.distplot(filtered_data, ax=ax, label=estimator_id, color=color, rug=False, kde=True, bins=64,
                                 hist_kws={"alpha": 0.3}, kde_kws={"alpha": 0.8})

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

        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(logdir, "gradients-dist.png"))
        #plt.show(block=False)
        #plt.pause(3)
        plt.close()


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
