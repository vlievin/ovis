import argparse
import json
import os
from shutil import rmtree

import pandas as pd
from booster.utils import logging_sep

from ovis.reporting.asymptotic import plot_statistics, plot_gradients_distribution
from ovis.reporting.style import *
from ovis.training.logging import get_loggers


def infer_parameter(configs, key):
    assert configs[key].nunique() == 1, f"Experiments have different values for the argument = `{key}`"
    return configs[key].unique()[0]


def read_and_report_asymptotic_experiment():
    """
    Read and report the results from `asymptotic_variance.py`. Gather all results from the `--exp` directory. Usage
    ```bash
    python report_asymptotic_variance.py --exp asymptotic-variance
    ```
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/', help='experiment directory')
    parser.add_argument('--output', default='reports/', help='output directory')
    parser.add_argument('--exp', default='asymptotic-variance', type=str, help='experiment id')
    parser.add_argument('--filter', default='', type=str, help='filter pattern')
    parser.add_argument('--latex', action='store_true', help='print as latex table')
    parser.add_argument('--float_format', default=".2f", help='float format')
    parser.add_argument('--draw_individual', action='store_true',
                        help='draw SNR and Variance independently for each parameter.')
    opt = vars(parser.parse_args())

    # get path to the experiment directory
    path = os.path.join(opt['root'], opt['exp'])
    experiments = [e for e in os.listdir(path) if '.' != e[0]]
    assert len(experiments) > 0, "No experiment found."

    # prepare output directory
    output_path = os.path.join(opt['output'], opt['exp'])
    if os.path.exists(output_path):
        rmtree(output_path)
    os.makedirs(output_path)

    # log console output to file
    logger, *_ = get_loggers(output_path, keys=['report'], format="%(message)s")
    logger.info(f"\n{logging_sep('=')}\nWriting figures to {os.path.abspath(output_path)}\n{logging_sep('=')}")

    """read data"""
    df = None
    grads = None
    configs = None
    for e in experiments:
        exp_dir = os.path.join(path, e)
        if len(opt['filter']) and opt['filter'] in e:
            print(f"-- Filtered:", e)
        elif "data.csv" not in os.listdir(exp_dir):
            print(f"-- not completed:", e)
        else:
            # read grads. stats.
            e_df = pd.read_csv(os.path.join(exp_dir, "data.csv"))

            # concat df
            if df is None:
                df = e_df
            else:
                df = pd.concat([df, e_df])

            # read raw gradients
            e_grads = pd.read_csv(os.path.join(exp_dir, "grads.csv"))
            if grads is None:
                grads = e_grads
            else:
                grads = pd.concat([grads, e_grads])

            # read config
            with open(os.path.join(exp_dir, 'config.json'), 'r') as fp:
                conf = json.load(fp)
                conf = pd.DataFrame([conf])
            if configs is None:
                configs = conf
            else:
                configs = pd.concat([configs, conf])

    assert df is not None, "No experiment could be parsed."

    # infer global parameters from the individual configurations
    opt['key_filter'] = infer_parameter(configs, "key_filter")

    # format estimator names
    df['estimator'] = list(map(format_estimator_name, df['estimator'].values))
    grads['estimator'] = list(map(format_estimator_name, grads['estimator'].values))

    """plotting"""
    set_matplotlib_style()
    plot_statistics(df, opt, output_path)
    if len(grads):
        plot_gradients_distribution(grads, output_path)


if __name__ == '__main__':
    read_and_report_asymptotic_experiment()
