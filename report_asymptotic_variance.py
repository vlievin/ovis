import argparse
import json
import os
from datetime import datetime
from shutil import rmtree

import pandas as pd

from ovis.plotting.style import *
from ovis.plotting.variance_plotting import plot_statistics, plot_gradients_distribution
from ovis.training.logging import get_loggers

# set custom plot style
set_style()

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='experiment directory')
parser.add_argument('--output', default='reports/', help='output directory')
parser.add_argument('--exp', default='asymptotic-variance', type=str, help='experiment id')
parser.add_argument('--filter', default='', type=str, help='filter pattern')
parser.add_argument('--latex', action='store_true', help='print as latex table')
parser.add_argument('--float_format', default=".2f", help='float format')
parser.add_argument('--draw_individual', action='store_true',
                    help='draw statistics for parameters individually (as in the original exp.)')
opt = parser.parse_args()

_sep = os.get_terminal_size().columns * "-"

# get path to the experiment directory
path = os.path.join(opt.root, opt.exp)
experiments = [e for e in os.listdir(path) if '.' != e[0]]

# prepare output directory
_id = opt.exp
output_path = os.path.join(opt.output, _id)
if os.path.exists(output_path):
    rmtree(output_path)
os.makedirs(output_path)

# log console output to file
logger, *_ = get_loggers(output_path, keys=['report'], format="%(message)s")
logger.info(f"{datetime.now()}\n\n")


# utilities
def _print_df(df):
    if opt.latex:
        logger.info(df.to_latex(float_format=f"%{opt.float_format}"))
    else:
        logger.info(df)


def infer_parameter(configs, key):
    assert configs[key].nunique() == 1, f"Experiments have different values for the argument = `{key}`"
    return configs[key].unique()[0]


"""read data"""
df = None
grads = None
configs = None
for e in experiments:

    print(">>>>", e)
    exp_dir = os.path.join(path, e)
    if opt.filter not in e and "data.csv" in os.listdir(exp_dir):
        # read grads. stats.
        e_df = pd.read_csv(os.path.join(exp_dir, "data.csv"))
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

"""post-processing"""
# infer parameters from individual configs
opt.key_filter = infer_parameter(configs, "key_filter")
opt.iw_oracle = infer_parameter(configs, "iw_oracle")
opt.oracle = infer_parameter(configs, "oracle")

# format estimator names
df['estimator'] = list(map(format_estimator_name, df['estimator'].values))
grads['estimator'] = list(map(format_estimator_name, grads['estimator'].values))

"""plotting"""
plot_statistics(df, opt, output_path)

if len(grads):
    plot_gradients_distribution(grads, output_path)
