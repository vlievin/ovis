"""original code at https://github.com/vmasrani/tvo/blob/master/discrete_vae/measure_efficiency.py"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from booster.utils import available_device
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from ovis.datasets import get_binmnist_datasets
from ovis.estimators.config import parse_estimator_id
from ovis.reporting.legend import Legend
from ovis.reporting.plotting import PLOT_WIDTH, PLOT_HEIGHT, ESTIMATOR_STYLE, update_labels
from ovis.reporting.style import DPI, MARKERS, METRIC_DISPLAY_NAME, set_matplotlib_style, format_estimator_name
from ovis.reporting.utils import sort_estimator_keys
from ovis.training.arguments import add_base_args, add_iw_sweep_args, add_model_architecture_args
from ovis.training.initialization import init_logging_directory, init_model
from ovis.training.logging import get_loggers
from ovis.training.ops import training_step
from ovis.training.utils import get_hash_from_opt, preprocess
from ovis.utils.success import Success
from ovis.utils.utils import Header, print_info


def main():
    parser = argparse.ArgumentParser()
    add_base_args(parser, exp="efficiency", dataset="binmnist")
    add_iw_sweep_args(parser, min=5, max=3e3, steps=10)
    add_model_architecture_args(parser)
    parser.add_argument('--load', default='',
                        help='existing experiment path to load from')
    parser.add_argument('--num_runs', default=5, type=int,
                        help='number of runs')
    parser.add_argument('--max_epoch_length', default=1e9, type=int,
                        help='maximum number of epochs per run')

    # estimators
    parser.add_argument('--estimators',
                        default='vimco-arithmetic,ovis-gamma1,reinforce,tvo-part2-config1',
                        help='comma separated list of estimators')
    parser.add_argument('--bs', default=24, type=int,
                        help='batch size')
    parser.add_argument('--filter',
                        default='',
                        help='filter estimator when plotting')

    opt = vars(parser.parse_args())

    # defining the run identifier
    deterministic_id = get_hash_from_opt(opt)
    run_id = f"efficiency-{opt['estimators']}-iw{opt['iw_min']}-{opt['iw_max']}-{opt['iw_steps']}-seed{opt['seed']}"
    if opt['exp'] != "":
        run_id += f"-{opt['exp']}"
    run_id += f"{deterministic_id}"
    exp_id = f"efficiency-{opt['exp']}-{opt['seed']}"

    # number of particles
    iws = [int(k) for k in np.geomspace(start=opt['iw_min'], stop=opt['iw_max'], num=opt['iw_steps'])[::-1]]

    # estimator ids
    estimator_ids = opt['estimators'].replace(" ", "").split(",")

    # defining the run directory
    logdir = init_logging_directory(opt, run_id)

    # save run configuration to the log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fp:
        opt['hash'] = hash
        fp.write(json.dumps(opt, default=lambda x: str(x), indent=4))

    # wrap the training loop inside with `Success` to write the outcome of the run to a file
    with Success(logdir=logdir):

        if opt['load'] == '':
            # get the device (cuda/cpu)
            device = available_device()
            assert 'cuda' in device, "No CUDA device detected."

            # define logger
            base_logger, *_ = get_loggers(logdir, keys=[exp_id])
            print_info(logdir=logdir, device=device, run_id=run_id, logger=base_logger)

            # setting the random seed
            torch.manual_seed(opt['seed'])
            np.random.seed(opt['seed'])

            # dataset & loader
            assert opt['dataset'] == 'binmnist', "Only implemented for Binarized MNIST"
            dset, *_ = get_binmnist_datasets(opt['data_root'], transform=ToTensor())
            loader = DataLoader(dset, batch_size=opt['bs'], shuffle=True, num_workers=1)

            # model
            model, hyperparameters = init_model(opt, dset[0], loader)
            model.to(device)
            model.train()

            # optimizer
            optimizer = Adam(model.parameters(), lr=1e-3)

            data = []
            iter_per_epoch = min(opt['max_epoch_length'], -(-len(dset) // opt['bs']))
            num_of_iterations = len(estimator_ids) * len(iws) * opt['num_runs'] * iter_per_epoch
            pbar = tqdm(total=num_of_iterations)
            for e, estimator_id in enumerate(estimator_ids):
                for i, iw in enumerate(iws):
                    # estimator
                    Estimator, config = parse_estimator_id(estimator_id)
                    estimator = Estimator(baseline=None, mc=1, iw=iw, **config)
                    estimator.to(device)

                    for run_i in range(opt['num_runs']):
                        pbar.set_description(
                            f"{estimator_id} [{e + 1}/{len(estimator_ids)}], "
                            f"K={iw} [{i + 1}/{len(iws)}], "
                            f"[{run_i + 1}/{opt['num_runs']}]")

                        # reset trackers
                        torch.cuda.reset_max_memory_allocated(device=device)
                        start = time.time()

                        # training epoch
                        for step, batch in enumerate(loader):
                            x, y = preprocess(batch, device)
                            training_step(x, model, estimator, [optimizer], y=y, return_diagnostics=False)
                            pbar.update(1)
                            if step >= opt['max_epoch_length']:
                                break

                        # end trackers
                        elapsed_time = time.time() - start
                        max_memory = torch.cuda.max_memory_allocated(device=device) / 1e6

                        # store data
                        data += [{
                            'estimator': estimator_id,
                            'iw': iw,
                            'run_i': run_i,
                            'max_memory': max_memory,
                            'elapsed_time': elapsed_time
                        }]

            # compile data into DataFrame and save to .csv
            data = pd.DataFrame(data)
            data.to_csv(os.path.join(logdir, 'efficiency.csv'))
        else:
            data = pd.read_csv(os.path.join(opt['load'], 'efficiency.csv'))

        data['estimator'] = data['estimator'].map(format_estimator_name)

        # plotting
        set_matplotlib_style()
        keys = ['max_memory', 'elapsed_time']
        fig, axes = plt.subplots(nrows=1, ncols=len(keys),
                                 figsize=(2 * PLOT_WIDTH, 2 * PLOT_HEIGHT),
                                 dpi=DPI)
        hue_order = list(data['estimator'].unique())
        if opt['filter'] != "":
            hue_order = [e for e in hue_order if opt['filter'] not in e]
        sort_estimator_keys(hue_order)

        legend = Legend(fig)
        for ax, key in zip(axes, keys):

            for e, estimator in enumerate(hue_order):
                sub_df = data[data['estimator'] == estimator]

                # color and marker
                if estimator in ESTIMATOR_STYLE:
                    style = {'color': ESTIMATOR_STYLE[estimator]['color'],
                             'marker': ESTIMATOR_STYLE[estimator]['marker']}
                else:
                    style = {'color': sns.color_palette()[e], 'marker': MARKERS[e]}

                # extract mean and std
                series = sub_df[['iw', key]].groupby('iw').agg(['mean', 'std'])
                series.reset_index(inplace=True)

                # area plot for CI
                ax.fill_between(series['iw'], series[key]['mean'] - 0.5 * series[key]['std'],
                                series[key]['mean'] + 0.5 * series[key]['std'], color=style['color'],
                                alpha=0.2)

                # plot mean value
                ax.plot(series['iw'], series[key]['mean'], markersize=0, alpha=0.5, **style)
                ax.plot(series['iw'], series[key]['mean'], label=estimator, alpha=1, **style)

            # labels and axis scale
            ax.set_yscale('log', basey=10)
            ax.set_xscale('log', basex=10)
            ax.set_ylabel(key)
            ax.set_xlabel("iw")
            legend.update(ax)

        update_labels(axes, METRIC_DISPLAY_NAME)
        legend.draw(group=True)
        plt.savefig(os.path.join(logdir, "efficiency.png"))
        plt.close()

        with Header(f"Data [Logging Directory: {os.path.abspath(logdir)} ]"):
            print(data.pivot_table(values=['max_memory', 'elapsed_time'], index=['iw', 'estimator'], aggfunc=np.mean))


if __name__ == '__main__':
    main()
