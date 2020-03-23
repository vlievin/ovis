import argparse
import json
import os
import traceback
from collections import defaultdict
from datetime import datetime
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotmap import DotMap
from tqdm import tqdm

from lib.logging import get_loggers
from lib.utils import parse_numbers

sns.set()

try:
    from tbparser.summary_reader import SummaryReader
    from tbparser import EventsFileReader
except:
    print("You probably need to install tbparser:\n   pip install git+https://github.com/velikodniy/tbparser.git")
    exit()

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='experiment directory')
parser.add_argument('--output', default='reports/', help='output directory')
parser.add_argument('--exp', default='exclude-sample-0.1', type=str, help='experiment id')
parser.add_argument('--filter', default='', type=str, help='filter run by id')
parser.add_argument('--pivot_metrics', default='max:elbo, avg:log_grad_var', type=str,
                    help='comma separated list of metrics to report in the table, the prefix defines the aggregation function (min, avg, max)')
parser.add_argument('--metrics',
                    default='train:control_variate_l1, train:log_grad_var, train:elbo, train:N_eff, train:nll, train:kl',
                    type=str,
                    help='comma separated list of keys to read from the tensorboard logs')
parser.add_argument('--ylims', default='', type=str,
                    help='comma separated list of limit values for the curve plot (syntax: key:min:max), example: `elbo:-60:-5,kl:4:10`')
parser.add_argument('--main_key', default='estimator', type=str, help='main parameter to include in the report')
parser.add_argument('--aux_key', default='iw', type=str, help='auxiliary parameter to include in the report')
parser.add_argument('--third_key', default='', type=str, help='second auxiliary parameter to include in the report')
parser.add_argument('--parse_estimator_args', action='store_true', help='parse estimator arguments such that `vimco-outer` -> `vimco` + outer=True')
parser.add_argument('--latex', action='store_true', help='print as latex table')
parser.add_argument('--float_format', default=".2f", help='float format')
parser.add_argument('--nsamples', default=64, type=int, help='number of points in the line plot')
opt = parser.parse_args()

_sep = os.get_terminal_size().columns * "-"

# get path to the experiment directory
path = os.path.join(opt.root, opt.exp)
experiments = [e for e in os.listdir(path) if '.' != e[0]]

# prepare output diorectory
_id = opt.exp
if len(opt.filter):
    _id += f"-filter{opt.filter}"
output_path = os.path.join(opt.output, _id)
if os.path.exists(output_path):
    rmtree(output_path)
os.makedirs(output_path)

# log console output to file
logger, *_ = get_loggers(output_path, keys=['report'], format="%(message)s")
logger.info(f"{datetime.now()}\n\n")


# utilities
def _to_tb(key):
    return f"loss/{key}"


def _from_tb(key):
    return key.replace("loss/", "")


def _print_df(df):
    if opt.latex:
        logger.info(df.to_latex(float_format=f"%{opt.float_format}"))
    else:
        logger.info(df)


# define keys to read from the logs ("header" is the tensorboard key (train/valid), "key" is the value (elbo, nll))
_headers, _metrics = zip(*[u.split(":") for u in opt.metrics.replace(" ", "").split(',')])
__metrics = defaultdict(list)
[__metrics[h].append(k) for (h, k) in zip(_headers, _metrics)]
tf__metrics = {h: list(map(_to_tb, v)) for h, v in __metrics.items()}

metrics_agg_fns, metrics = zip(*[u.split(':') for u in opt.pivot_metrics.replace(" ", "").split(",")])
metrics_agg_fns = [{'min': np.min, 'max': np.max, 'avg': np.mean}[m] for m in metrics_agg_fns]
metrics = list(metrics)

# filters
filters = opt.filter.replace(" ", "").split(',') if len(opt.filter) else ""

print("# Metrics:", metrics)
print("# Agg. Fns:", metrics_agg_fns)

# read data
logger.info(f"# reading experiments from path: {path}")
logger.info(_sep)
data = []  # store hyperparameters and configs
logs = []  # store training data from tensorboard logs
pbar = tqdm(experiments)
for e in pbar:
    pbar.set_description(f"{e}")

    exp_path = os.path.join(path, e)
    files = os.listdir(exp_path)
    _success_file = 'success.txt'
    try:
        if not _success_file in files:
            logger.info(f" >>>  Not yet completed: {e}")
        elif any([(u in e) for u in filters]):
            logger.info(f" >>>  filtered: {e}")
        else:
            with open(os.path.join(exp_path, _success_file), 'r') as fp:
                success_ = fp.read()

            if not "Success." in success_:
                logger.info(f" >>> Failed: {e}")
            else:
                # reading configuration files with run parameter
                with open(os.path.join(exp_path, 'config.json'), 'r') as fp:
                    args = DotMap(json.load(fp))

                # parse estimator args: e.g.
                # * `vimco-outer` -> estimator=vimco, outer=True
                # * `vimco-z_reject6` -> estimator=vimco, z_reject=6
                if opt.parse_estimator_args and "-" in args.estimator:

                    estimator_args = args.estimator.split("-")

                    # estimator is the first arg
                    args['estimator'] = estimator_args[0]

                    # for each estimator arg
                    for arg in estimator_args[1:]:
                        if arg == "geometric":
                            args.estimator = args.estimator.replace(f"-geometric", "")
                            args["w_hat"] = "geometric"
                        elif arg == "arithmetic":
                            args.estimator = args.estimator.replace(f"-arithmetic", "")
                            args["w_hat"] = "arithmetic"
                        else:
                            numbers = parse_numbers(arg)
                            if len(numbers):
                                value = numbers[0]
                                arg = arg.replace(str(value), "")
                                args[arg] = value
                            else:
                                args[arg] = True

                # read tensorboard logs
                for header in _headers:
                    _dir = os.path.join(exp_path, header)
                    _tf_log = [os.path.join(_dir, o) for o in os.listdir(_dir) if 'events.out.tfevents' in o][0]
                    with open(_tf_log, 'rb') as f:
                        reader = EventsFileReader(f)

                        for item in reader:
                            step = item.step
                            for v in item.summary.value:
                                if v.tag in tf__metrics[header]:
                                    logs += [{'id': e, 'step': step, '_key': _from_tb(v.tag),
                                              '_value': float(v.simple_value)}]

                # gather config/hyperparameters
                d = dict(args)
                d['id'] = e
                # append data and logs
                data += [d]

    except Exception as ex:
        logger.info("## FAILED. Exception:")
        logger.info(_sep)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        logger.info(_sep)
        logger.info("\nException: ", ex, "\n")

# exit if not data
if len(data) == 0:
    logger.info(
        f"{_sep}\nCouldn't read any record. Either they are errors or the experiments are not yet completed.\n{_sep}")
    exit()

# compile data into a dataframe
df = pd.DataFrame(data)

# replace void estimator arguments with False (important since nans are dropped afterwards)
# for instance, for thwo different estimator names
# *  "vimco-zreject6" is parsed into estimator = vimco, zreject=6
# *  "vimco" is parsed into estimator = vimco, zreject=None
df = df.fillna(0)  # todo: fill with zero when numbers else False

# drop columns that contain the same attributes (they will be dropped from `df`)
nunique = df.apply(pd.Series.nunique)
global_attributes = list(nunique[nunique == 1].index)
# remove some metrics from the global attributes if they have to be indexed from `df` later on.
for k in metrics + ['seed', opt.aux_key]:
    if k in global_attributes:
        global_attributes.remove(k)
df = df.drop(global_attributes, axis=1)

# compile log data and merge with the attributes
logs = pd.DataFrame(logs)

# join df.config into logs
_keys_to_merge = [k for k in df.keys() if k not in metrics]
logs = logs.merge(df[_keys_to_merge], left_on="id", right_on="id")

# aggregate `metric` over logs and join into `df`
for m, agg_fn in zip(metrics, metrics_agg_fns):
    logs_m = logs[logs["_key"] == m]
    agg_log_m = pd.DataFrame(logs_m[['id', "_value"]].groupby('id')["_value"].apply(agg_fn))
    agg_log_m.rename(columns={'_value': m}, inplace=True)
    df = df.merge(agg_log_m, left_on="id", right_on="id")

# drop id (exp name)
df = df.drop('id', 1)
logs = logs.drop('id', 1)

"""
print all results
"""

# sort by the first metric
df = df.sort_values(metrics[0], ascending=False)

# print all results
logger.info("\n" + _sep)
for g in global_attributes:
    logger.info(f"{g} : {args[g]}")
logger.info(_sep)
logger.info(os.path.abspath(path))
logger.info(_sep)
logger.info(f"all data: varying parameters: {[k for k in df.keys() if k not in metrics]}")
logger.info(_sep)
_print_df(df)

"""
pivot table
"""


def aggfunc(serie):
    mean = np.mean(serie)
    std = np.std(serie)
    return f"{mean:{opt.float_format}} +/- {std:{opt.float_format}} (n={len(serie)})"


_columns = []  # [opt.aux_key] if len(opt.aux_key) > 0 else []
_index = [k for k in df.keys() if k not in metrics + _columns + ["seed", opt.main_key, opt.aux_key]] + [opt.main_key]
if len(opt.aux_key):
    _index = [opt.aux_key] + _index
pivot = df.pivot_table(index=_index, columns=_columns, values=metrics, aggfunc=aggfunc)

# sort pivot according to the mean value
mean_pivot = df.pivot_table(index=_index, values=metrics[0], aggfunc=np.mean)
mean_pivot = mean_pivot.sort_values(metrics[0], ascending=False)
# mean_pivot.sort_values([*_index, metrics[0]], ascending=True, inplace=True)
pivot = pivot.reindex(mean_pivot.index)

# sort index
for idx in _index[::-1][1:]:
    pivot.sort_index(level=idx, sort_remaining=False, inplace=True)

logger.info("\n" + _sep)
logger.info("Pivot table:\n" + _sep)
_print_df(pivot)
logger.info(_sep)

# save to file
df.to_csv(os.path.join(output_path, "pivot.csv"))

"""
plot line plots with uncertainty intervals
"""

_last_indexes = ['seed', '_key', 'step']
_index = [k for k in logs.keys() if k != '_value' and k not in _last_indexes]
_index += _last_indexes

# get max step
M = logs['step'].max()
bins = [int(s) for s in range(0, M, M // opt.nsamples)]

n_full = len(logs['step'].unique())
ratio = n_full // opt.nsamples

# reshape data with index [..., seed, _key, step] and sort by steps
# logs = logs.pivot_table(index=_index, values='_value', aggfunc=np.mean)
# for idx in _index[::-1]:
#     logs.sort_index(level=idx, sort_remaining=False, inplace=True)

logger.info(f"Downsampling data.. (n={opt.nsamples})")
logs.reset_index(level=-1, inplace=True)
_index.remove("step")
logs = logs.groupby(_index + [pd.cut(logs.step, bins)]).mean()
logs.index.rename(level=[-1], names=['step_bucket'], inplace=True)
logs.reset_index(inplace=True)

# drop nan
logs.dropna(inplace=True)

# save to file
logs.to_csv(os.path.join(output_path, "curves.csv"))


def plot_logs(logs, _keys, path, style_key=None, ylims=dict()):
    """
    Make a grid of line plots with std intervals.
    :param logs: dataframe containing all the training data per time step
    :param _keys: keys to plot (elbo, nll, ...)
    :param path: output file
    :return: None
    """

    N = len(_keys)
    ncols = 2
    nrows = N // ncols

    if N > ncols * nrows:
        nrows += 1
        _keys = list(_keys) + [_keys[-1] for _ in
                               range((ncols * nrows - len(_keys)))]  # repeat the last plot for the legend

    hue_order = list(logs[opt.main_key].unique())
    step_min = np.percentile(logs['step'].values.tolist(), 10)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    for i, k in tqdm(list(enumerate(_keys))):
        u = i // ncols
        v = i % ncols
        ax = axes[u, v]

        sns.lineplot(x="step", y="_value",
                     hue=opt.main_key,
                     hue_order=hue_order,
                     style=style_key,
                     data=logs[logs['_key'] == k], ax=ax,
                     # palette=sns.color_palette("mako_r", 4)
                     )

        ax.set_ylabel(k)
        # y lims
        if k in ylims:
            ax.set_ylim(ylims[k])
        else:
            ys = logs[(logs['_key'] == k) & (logs['step'] > step_min)]['_value'].values.tolist()
            if len(ys):
                a, b = np.percentile(ys, [25, 75])
                M = b - a
                k = 1.5
                ax.set_ylim([a - k * M, b + k * M])

        if i < len(_keys) - 1:
            ax.get_legend().remove()

    # draw legend in the last plot
    # patches = [mpatches.Patch(color=sns.color_palette()[i], label=key) for i, key in enumerate(hue_order)]
    # axes[nrows - 1, ncols - 1].legend(handles=patches)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# plot for all auxiliary keys
if len(opt.ylims):
    ylims = [u.split(":") for u in opt.ylims.replace(" ", "").split(',')]
    ylims = {u[0]: [eval(u[1]), eval(u[2])] for u in ylims}
else:
    ylims = {}
logger.info("Generating merged plots ...")
plot_logs(logs, _metrics, os.path.join(output_path, f"curves.png"), ylims=ylims,
          style_key=opt.aux_key if len(opt.aux_key) else None)
if len(opt.aux_key):
    # on plot for each key
    values = list(logs[opt.aux_key].unique())
    for i, v in enumerate(sorted(values)):
        f"Generating plots for {opt.aux_key} = {v} [{i + 1} / {values}]"
        plot_logs(logs[logs[opt.aux_key] == v], _metrics, os.path.join(output_path, f"curves-{opt.aux_key}={v}.png"),
                  ylims=ylims, style_key=opt.third_key if len(opt.third_key) else None)
