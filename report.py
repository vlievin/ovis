import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from shutil import rmtree

import pandas as pd
from dotmap import DotMap

from lib.logging import get_loggers
from lib.plotting import *
from lib.utils import parse_numbers

sns.set(style="whitegrid")
sns.set(style="ticks")
sns.set_context("paper", font_scale=1.2)

try:
    from tbparser.summary_reader import SummaryReader
    from tbparser import EventsFileReader
except:
    print("You probably need to install tbparser:\n   pip install git+https://github.com/velikodniy/tbparser.git")
    exit()

log_rules = {
    'loss/elbo': 'logx',
    'loss/kl': 'logx',
    'loss/nll': 'logx',
    'loss/r_eff': 'logx',
    'grads/snr': 'loglog',
    'grads/dsnr': 'loglog',
    'grads/variance': 'loglog',
    'grads/magnitude': 'loglog',
    'grads/direction': 'logx',
    'gmm/posterior_mse': 'loglog',
    'gmm/prior_mse': 'loglog'
}

metric_dict = {
    'iw' : r"$K$",
    'c_iw' : r"$K$",
    'loss/elbo': r"$\mathcal{L}_K$",
    'loss/kl': r"$KL(q_{\phi}(z | x) | p(z))$",
    'loss/nll': r"$- \log p_{\theta}(z | x)$",
    'loss/r_eff': r"$ESS / K$",
    'grads/variance': r"$Var(\Delta_K(\phi))$",
    'grads/snr': r"$SNR(\Delta_K(\phi))$",
    'grads/dsnr': r"$DSNR(\Delta_K(\phi))$",
    'grads/magnitude': r"$ | E[\Delta_K(\phi)] | $",
    'grads/direction': r"$cosine( \Delta_K(\phi) ,  \Delta_K^{oracle}(\phi) )$",
    'gmm/posterior_mse': r"$\left\| q_{\phi}(z | x) - p_{\theta_{true}}(z | x) \right\| $",
    'gmm/prior_mse': r"$\left\| p_{\theta}(z) - p_{\theta_{true}}(z) \right\| $"

}

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='experiment directory')
parser.add_argument('--output', default='reports/', help='output directory')
parser.add_argument('--exp', default='exclude-sample-0.1', type=str, help='experiment id')
parser.add_argument('--include', default='', type=str, help='filter run by id')
parser.add_argument('--exclude', default='', type=str, help='filter run by id')
parser.add_argument('--pivot_metrics', default='max:train:loss/elbo, avg:train:grads/log_snr, avg:train:loss/r_eff',
                    type=str,
                    help='comma separated list of metrics to report in the table, the prefix defines the aggregation function (min, avg, max)')
parser.add_argument('--metrics',
                    default='train:reinforce/l1, train:grads/log_snr, train:loss/elbo, train:loss/r_eff, train:loss/nll, train:loss/kl',
                    type=str,
                    help='comma separated list of keys to read from the tensorboard logs')
parser.add_argument('--ylims', default='', type=str,
                    help='comma separated list of limit values for the curve plot (syntax: key:min:max), example: `elbo:-60:-5,kl:4:10`')
parser.add_argument('--keys', default='estimator, iw', type=str,
                    help='comma separated list of keys to include in the report by decreasing order of importance, more than 3 keys is not yet handled for plotting.')
parser.add_argument('--spot_on_metrics', default='train:grads/log_snr, train:loss/elbo', type=str,
                    help='comma separated list of keys for the `spot-on` plots. ')
parser.add_argument('--parse_estimator_args', action='store_true',
                    help='parse estimator arguments such that `vimco-outer` -> `vimco` + outer=True')
parser.add_argument('--counterfactuals', action='store_true',
                    help='create on run for each counterfactual record')
parser.add_argument('--latex', action='store_true', help='print as latex table')
parser.add_argument('--float_format', default=".2f", help='float format')
parser.add_argument('--nsamples', default=64, type=int, help='target number of points in the line plot (downsampling)')
parser.add_argument('--non_completed', action='store_true', help='also keep runs that are not yet completed.')
parser.add_argument('--max_records', default=-1, type=int,
                    help='only read the first `max_records` data point (`-1` = no limit)')
parser.add_argument('--skip_level', default=-1, type=int,
                    help='skip nesting levels while plotting')
parser.add_argument('--merge_args', default='', type=str, help='list of args to merge into one')
opt = parser.parse_args()

_sep = os.get_terminal_size().columns * "-"

# get path to the experiment directory
path = os.path.join(opt.root, opt.exp)
experiments = [e for e in os.listdir(path) if '.' != e[0]]

# prepare output diorectory
_id = opt.exp
if len(opt.include):
    _id += f"-inc={opt.include}"
if len(opt.exclude):
    _id += f"-exc={opt.exclude}"
if opt.counterfactuals:
    _id += f"-counterfactuals"
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


def get_counterfactuals_headers(exp_path):
    headers = (p for p in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, p)))
    return [p for p in headers if p not in ['train', 'valid', 'test']]


def read_tf_record(exp_path, exp, dict_metrics, force_header=None, meta=dict()):
    for header in dict_metrics.keys():
        _dir = os.path.join(exp_path, header)
        _tf_log = [os.path.join(_dir, o) for o in os.listdir(_dir) if 'events.out.tfevents' in o][0]
        with open(_tf_log, 'rb') as f:
            reader = EventsFileReader(f)
            for item in reader:
                step = item.step
                for v in item.summary.value:
                    if v.tag in dict_metrics[header]:
                        if force_header is not None:
                            _header = force_header
                        else:
                            _header = header
                        yield {'id': exp, 'step': step, '_key': f"{_header}:{v.tag}", '_value': float(v.simple_value),
                               'header': header, **meta}


# define the metrics for the grid of line plots
curves_metrics = opt.metrics.replace(" ", "").split(',')

# define the metrics for the `spot-on` plot
keys = list(opt.keys.replace(" ", "").split(','))
spot_on_metrics = list(opt.spot_on_metrics.replace(" ", "").split(','))
if spot_on_metrics[0] == "":
    spot_on_metrics = []


# define the `pivot table` metrics
def _parse_pivot_metric(u):
    """a metric is expressed as a `:` separated string with syntax `agg_fn:header:key`, e.g. `avg:train:loss/elbo`"""
    agg_fn, header, key = u.split(":")
    agg_fn = {'min': np.min, 'max': np.max, 'avg': np.mean}[agg_fn]
    return agg_fn, f"{header}:{key}"


pivot_metrics_agg_fns, pivot_metrics = zip(
    *[_parse_pivot_metric(u) for u in opt.pivot_metrics.replace(" ", "").split(",")])
pivot_metrics = list(pivot_metrics)

# define the metrics to read from tensorboard
all_metrics = list(set(curves_metrics + spot_on_metrics + pivot_metrics))
_headers, _all_metrics = zip(*[u.split(":") for u in all_metrics])
dict_metrics = defaultdict(list)
[dict_metrics[h].append(k) for (h, k) in zip(_headers, _all_metrics)]

# filters
filters_inc = opt.include.replace(" ", "").split(',') if len(opt.include) else ""
print("Filters include:", filters_inc)
filters_exc = opt.exclude.replace(" ", "").split(',') if len(opt.exclude) else ""
print("Filters exclude", filters_exc)


def is_filtered(exp, filters_inc, filters_exc):
    return any([(u in exp) for u in filters_exc]) or any([(u not in exp) for u in filters_inc])


_success_file = 'success.txt'


def is_successful(exp_path):
    with open(os.path.join(exp_path, _success_file), 'r') as fp:
        success_ = fp.read()

    return "Success." in success_


print("# Metrics:", all_metrics)
print("# Agg. Fns:", pivot_metrics_agg_fns)

# read data
logger.info(f"# reading experiments from path: {path}")
logger.info(_sep)
data = []  # store hyperparameters and configs
logs = []  # store training data from tensorboard logs
counterfactual_logs = []  # store logs form counterfactuals
pbar = tqdm(experiments)
for e in pbar:
    pbar.set_description(f"{e}")

    exp_path = os.path.join(path, e)
    files = os.listdir(exp_path)
    try:
        if (not opt.non_completed) and (not _success_file in files):
            logger.info(f" >>>  Not yet completed: {e}")
        elif is_filtered(e, filters_inc, filters_exc):
            logger.info(f" >>>  filtered: {e}")
        else:
            if not opt.non_completed and not is_successful(exp_path):
                logger.info(f" >>> Failed: {e}")
            else:
                # reading configuration files with run parameter
                with open(os.path.join(exp_path, 'config.json'), 'r') as fp:
                    args = DotMap(json.load(fp))
                    args.pop("hostname")
                    args.pop("root")  # todo

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
                logs += [r for r in read_tf_record(exp_path, e, dict_metrics)]

                # store counterfactuals data
                if opt.counterfactuals:
                    # build a ditctionary with keys=counterfactual id, values = dict_metrics['train']
                    c_dict_metrics = {k: dict_metrics['train'] for k in get_counterfactuals_headers(exp_path)}
                    counterfactual_logs += [r for r in
                                            read_tf_record(exp_path, e, c_dict_metrics, force_header='train')]

                # gather config/hyperparameters
                d = dict(args)
                d['id'] = e
                # append data and logs
                data += [d]

                if opt.max_records > 0 and len(data) > opt.max_records:
                    break

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
# *  "vimco" is parsed into estimator = vimco, zreject=0
df = df.fillna(0)  # todo: fill with zero when numbers else False

# merging
if opt.merge_args != "":
    a, b = opt.merge_args.replace(" ", "").split(",")
    merge_name = f"{a}-{b}"
    df[merge_name] = [f"{x}-{y}" for x, y in zip(df[a].values, df[b].values)]
    df = df.drop(a, 1)
    df = df.drop(b, 1)

# drop columns that contain the same attributes (they will be dropped from `df`)
nunique = df.apply(pd.Series.nunique)
global_attributes = [k for k in list(nunique[nunique == 1].index) if k not in ['id']]
# remove some metrics from the global attributes if they have to be indexed from `df` later on.
for k in pivot_metrics + ['seed', 'dataset'] + keys:
    if k in global_attributes:
        global_attributes.remove(k)

df = df.drop(global_attributes, axis=1)

# compile log data and merge with the attributes
logs = pd.DataFrame(logs)

# join df.config into logs
_keys_to_merge = [k for k in df.keys() if k not in pivot_metrics]
logs = logs.merge(df[_keys_to_merge], left_on="id", right_on="id", how='left')

# integrate counterfactuals data into the logs
if len(counterfactual_logs):

    counterfactual_logs = pd.DataFrame(counterfactual_logs)

    counterfactual_logs.dropna(inplace=True)

    # replace header by `train` and create new `counterfactual` column
    counterfactual_logs.rename(columns={'header': 'counterfactual'}, inplace=True)
    counterfactual_logs['header'] = 'train'


    # extract counter factual id
    def extract_iw(id, c, df):
        """extract iw number of c identifier if available else retrieve value from `df`"""
        last_split = c.split("-")[-1]
        splits = c.split("-")[:-1]
        if "iw" and not "iwae" in last_split:
            return "-".join(splits), eval(last_split.replace("iw", ""))
        else:
            return c, df[df['id'] == id]['iw'].values[0]


    counterfactual_logs['counterfactual'], counterfactual_logs['c_iw'] = zip(*[extract_iw(*u, df) for u in
                                                                               zip(counterfactual_logs['id'].values,
                                                                                   counterfactual_logs[
                                                                                       'counterfactual'].values)])

    # merge counterfactuals into the logs
    logs = counterfactual_logs.merge(logs, left_on=["id", "step", "header", "_key"],
                                     right_on=["id", "step", "header", "_key"], how='outer')


    # solve values conflict (i.e. keep counterfacutal data when available)
    def _merge_value(x, y):
        return x if x == x else y


    logs['_value'] = [_merge_value(*u) for u in zip(logs['_value_x'].values, logs['_value_y'].values)]
    logs.drop(['_value_x', '_value_y'], 1, inplace=True)

    # fill `c_iw` and `counterfactual` wen missing
    if 'estimator' in logs.keys():
        logs['counterfactual'].fillna(logs['estimator'], inplace=True)
    else:
        logs['counterfactual'].fillna(args['estimator'], inplace=True)
    if 'iw' in logs.keys():
        logs['c_iw'].fillna(logs['iw'], inplace=True)
    else:
        logs['c_iw'].fillna(args['iw'], inplace=True)

    # update `id` column for a transparent merge/groupby
    new_ids = defaultdict(list)
    # hack to get the list of values for _ids[0] and update `new_ids` in _ids[1] = None
    _ids = [(f"{i}-{e}-{iw}", new_ids[i].append(f"{i}-{e}-{iw}")) for i, e, iw in
            zip(*[logs[key] for key in ['id', 'counterfactual', 'c_iw']])]
    logs['id'] = list(zip(*_ids))[0]  # id-counterfactual-iw -> id
    # repeat each `id` in df with the values from `new_ids`
    new_df = []
    for idx, row in df.iterrows():
        row = dict(**row)
        old_id = row.pop('id')
        for new_id in new_ids[old_id]:
            # get c_iw and _estimator
            *counterfactual, c_iw = new_id.split('-')
            counterfactual = "-".join(counterfactual).replace(f"{old_id}-",
                                                              "")  # retrieve original name in a rather hacky way
            new_df += [{'id': new_id, 'counterfactual': counterfactual, 'c_iw': eval(c_iw, {'nan': 0}), **row}]
    df = pd.DataFrame(new_df)

    # drop header
    logs.drop('header', 1, inplace=True)

    # drop potential nans
    # print("-> DROP:", len(logs), len(df))
    # logs.dropna(inplace=True)
    # df.dropna(inplace=True)
    # print("---> DROP:", len(logs), len(df))

# aggregate `metric` over logs and join into `df`
for m, agg_fn in zip(pivot_metrics, pivot_metrics_agg_fns):
    logs_m = logs[logs["_key"] == m]
    agg_log_m = pd.DataFrame(logs_m[['id', "_value"]].groupby('id')["_value"].apply(agg_fn))
    agg_log_m.rename(columns={'_value': m}, inplace=True)
    df = df.merge(agg_log_m, left_on="id", right_on="id", how='right')

# drop id (exp name)
df.drop('id', 1, inplace=True)
logs.drop('id', 1, inplace=True)

"""
print all results
"""

# sort by the first metric
df = df.sort_values(pivot_metrics[0], ascending=False)

# print all results
logger.info("\n" + _sep)
for g in global_attributes:
    logger.info(f"{g} : {args[g]}")
logger.info(_sep)
logger.info(os.path.abspath(path))
logger.info(_sep)
logger.info(f"all data: varying parameters: {[k for k in df.keys() if k not in pivot_metrics]}")
logger.info(_sep)
_print_df(df)

"""
pivot table
"""


def aggfunc(serie):
    """function use to aggregate the results into the pivot table"""
    mean = np.mean(serie)
    std = np.std(serie)
    return f"{mean:{opt.float_format}} +/- {std:{opt.float_format}} (n={len(serie)})"


_columns = []  # [opt.aux_key] if len(opt.aux_key) > 0 else []
_excluded_keys = all_metrics + _columns + ["seed"] + keys[::-1]
_index = [k for k in df.keys() if k not in _excluded_keys]
_index = _index + keys[::-1]
# pivot table
pivot = df.pivot_table(index=_index, columns=_columns, values=pivot_metrics, aggfunc=aggfunc)

# build another pivot table using only the first metric to sort the original pivot table accordingly
mean_pivot = df.pivot_table(index=_index, values=pivot_metrics[0], aggfunc=np.mean)
mean_pivot = mean_pivot.sort_values(pivot_metrics[0], ascending=False)
# sort the original pivot table
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

# plot for all auxiliary keys
if len(opt.ylims):
    ylims = [u.split(":") for u in opt.ylims.replace(" ", "").split(',')]
    ylims = {u[0]: [eval(u[1]), eval(u[2])] for u in ylims}
else:
    ylims = {}

# todo: nested loop

_keys = keys
if logs['dataset'].nunique() > 1 and _keys[0] != 'dataset':  # TODO fix this
    _keys = ['dataset'] + _keys

print(_sep)
level = 1
print(f">>> Level = {level}, Plotting with keys:", _keys)
print(_sep)

# define keys used for styling
cat_key = _keys[0]
main_key = _keys[1]  # color
aux_key = _keys[2] if len(_keys) > 2 else None  # line style (in main plot)
third_key = _keys[3] if len(_keys) > 3 else None  # line style (in auxiliary plots)
fourth_key = _keys[4] if len(_keys) > 4 else None

meta = {'log_rules': log_rules, 'metric_dict': metric_dict}

if logs[cat_key].nunique() > 0:
    # pivot plot
    logger.info(f"|- Generating pivot plots with key = {cat_key} ..")
    _path = os.path.join(output_path, f"pivot-plot-all-level={level}-by={cat_key}-hue={main_key}.png")
    pivot_plot(df, _path, pivot_metrics, cat_key, main_key, aux_key, style_key=third_key, **meta)

    if cat_key != 'dataset':
        _path = os.path.join(output_path, f"curves-all-level={level}.png")
        plot_logs(logs, _path, curves_metrics, main_key, ylims=ylims, style_key=aux_key, **meta)

# plot all data for each key
level = 2
for cat in logs[cat_key].unique():
    print(_sep)
    logger.info(f"[{cat_key} = {cat}]")
    cat_logs = logs[logs[cat_key] == cat]
    cat_df = df[df[cat_key] == cat]

    logger.info(f"|- Generating pivot plots with key = {cat_key} ..")
    _path = os.path.join(output_path, f"{level}-pivot-plot-{cat_key}={cat}-by={cat_key}-hue={main_key}.png")
    pivot_plot(cat_df, _path, pivot_metrics, cat_key, main_key, aux_key, style_key=third_key, **meta)

    logger.info(f"|- Generating merged curves plots..")
    # main plot
    _path = os.path.join(output_path, f"{level}-curves-all-{cat_key}={cat}.png")
    plot_logs(cat_logs, _path, curves_metrics, main_key, ylims=ylims, style_key=aux_key, **meta)

    if aux_key is not None:

        level = 3

        # spot on plots
        if len(spot_on_metrics):
            logger.info(f"|- Generating spot-on plots for aux. key = {aux_key}")
            _path = os.path.join(output_path, f"{level}-{cat_key}={cat}-spot-on.png")
            spot_on_plot(cat_logs, _path, spot_on_metrics, main_key, aux_key, style_key=third_key, ylims=ylims, **meta)

        # on plot for each auxiliary key
        # for each aux key..
        aux_key_values = list(cat_logs[aux_key].unique())
        for i, v in enumerate(sorted(aux_key_values)):
            aux_cat_logs = cat_logs[cat_logs[aux_key] == v]
            logger.info(f"|--- Generating plots for {aux_key} = {v} [{i + 1} / {len(aux_key_values)}]")

            # auxiliary plot
            _path = os.path.join(output_path, f"{level}-{cat_key}={cat}-curves-{aux_key}={v}.png")
            plot_logs(aux_cat_logs, _path, curves_metrics, main_key, ylims=ylims, style_key=third_key, **meta)
