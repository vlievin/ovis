import json
import os
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from booster.utils import logging_sep
from tqdm import tqdm

from ovis.utils.utils import parse_numbers, BASE_ARGS_EXCEPTIONS, Success

try:
    from tbparser.summary_reader import SummaryReader
    from tbparser import EventsFileReader
except:
    print("You probably need to install tbparser:\n   pip install git+https://github.com/velikodniy/tbparser.git")
    exit()


def read_tf_record(exp_path, exp, dict_metrics, force_header=None, meta=dict()):
    """read individual Tensorboard records and index by `exp`"""
    for header in dict_metrics.keys():
        _dir = os.path.join(exp_path, header)
        for tf_log in [os.path.join(_dir, o) for o in os.listdir(_dir) if 'events.out.tfevents' in o]:
            with open(tf_log, 'rb') as f:
                reader = EventsFileReader(f)
                for item in reader:
                    step = item.step
                    for v in item.summary.value:
                        if v.tag in dict_metrics[header]:
                            if force_header is not None:
                                _header = force_header
                            else:
                                _header = header
                            yield {'id': exp, 'step': step, '_key': f"{_header}:{v.tag}",
                                   '_value': float(v.simple_value),
                                   'header': header, **meta}


def parse_pivot_metric(u):
    """a metric is expressed as a `:` separated string with syntax `agg_fn:header:key`, e.g. `avg:train:loss/elbo`"""
    agg_fn_id, header, key = u.split(":")
    return agg_fn_id, f"{header}:{key}"


def get_agg_fn(agg_fn_id):
    return {'min': np.min, 'max': np.max, 'avg': np.mean, 'mean': np.mean, 'last': lambda x: list(x)[-1]}[agg_fn_id]


def is_filtered(exp, filters_inc, filters_exc):
    """
    True if `exp` match pattern in the list `filters_exc`
    True if `exp` doesn't match pattern in the list `filters_inc`
    """
    exclude_cond = any([(u in exp) for u in filters_exc])
    include_cond = (len(filters_inc) > 0 and all([(u not in exp) for u in filters_inc]))
    return exclude_cond or include_cond


def is_successful(exp_path):
    with open(os.path.join(exp_path, Success.file), 'r') as fp:
        success_text = fp.read()

    return Success.success in success_text


def parse_estimator_args(args):
    """
    parse estimator args: e.g.
    * `ovis-gamma0` -> estimator=ovis, gamma=0.6
    :param args:
    :return:
    """
    estimator_args = args['estimator'].split("-")

    # estimator is the first arg
    args['estimator'] = estimator_args[0]

    # for each estimator arg
    for arg in estimator_args[1:]:
        numbers = parse_numbers(arg)
        if len(numbers):
            value = numbers[0]
            arg = arg.replace(str(value), "")
            args[arg] = value
        else:
            args[arg] = True

    return args


def drop_exceptions_from_args(args, exceptions=BASE_ARGS_EXCEPTIONS):
    """drop keys from `args`"""
    for e in exceptions:
        args.pop(e, None)
    return args


def format_estimator_name(name):
    """add special rules to format the estimator name"""
    if 'tvo' in name:
        return 'tvo'
    else:
        return name


def merge_args(data, args):
    """merge two args `X` and `Y` into `X-Y`"""
    a, b = args.replace(" ", "").split(",")
    merge_name = f"{a}-{b}"
    data[merge_name] = [f"{x}-{y}" for x, y in zip(data[a].values, data[b].values)]
    data = data.drop(a, 1)
    data = data.drop(b, 1)
    return data


def parse_comma_separated_list(u):
    """`x, y` -> [`x`, `y`]"""
    return u.replace(" ", "").split(',') if len(u) else []


def read_experiments(opt, metrics, logger):
    # get path to the experiment directory
    path = os.path.join(opt.root, opt.exp)
    experiments = [e for e in os.listdir(path) if '.' != e[0]]
    logger.info(f"{logging_sep('=')}\n# reading experiment at {path}\n{logging_sep('=')}")

    # parse filters from opt
    filters_inc, filters_exc = map(parse_comma_separated_list, [opt.include, opt.exclude])

    data = []  # store hyperparameters and configs
    logs = []  # store tensorboard logs
    pbar = tqdm(experiments)
    for e in pbar:
        pbar.set_description(f"{e}")
        exp_path = os.path.join(path, e)
        files = os.listdir(exp_path)
        try:
            if (not opt.non_completed) and (not Success.file in files):
                logger.info(f"   [Not completed] {e} ")
            elif is_filtered(e, filters_inc, filters_exc):
                logger.info(f"   [Filtered] {e}")
            else:
                if not opt.non_completed and not is_successful(exp_path):
                    logger.info(f"   [Failed] {e}")
                else:
                    # reading configuration files with run parameter
                    with open(os.path.join(exp_path, 'config.json'), 'r') as fp:
                        args = json.load(fp)

                    # remove exceptions and parse `estimator` ids
                    args = drop_exceptions_from_args(args)
                    args.pop('hash', None)
                    if opt.parse_estimator_args and "-" in args['estimator']:
                        args = parse_estimator_args(args)

                    # read tensorboard logs
                    logs += [r for r in read_tf_record(exp_path, e, metrics['all_metrics_by_header'])]

                    # gather config/hyperparameters
                    d = dict(args)
                    d['id'] = e
                    # append data and logs
                    data += [d]

                    # stop reading if > `max_records`
                    if opt.max_records > 0 and len(data) > opt.max_records:
                        break

        except Exception as ex:
            logger.info(f"    [Parsing Failed with Exception\n{logging_sep('=')}")
            logger.info(logging_sep("="))
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            logger.info(logging_sep("="))

    # exit if not data
    if len(data) == 0:
        logger.info(
            f"{logging_sep('=')}\nCouldn't read any record. Either they are errors or the experiments are not yet completed.\n{logging_sep('=')}")
        exit()

    # compile into a DataFrame
    data = pd.DataFrame(data)

    # replace NaN with False (important since nans are dropped afterwards)
    # NaNs arise when using `parse_estimator_args`
    data = data.fillna(False)

    # compile logs into a DataFrame
    logs = pd.DataFrame(logs)

    # check data
    found_keys = logs['_key'].unique()
    for metric in metrics['all_metrics']:
        if metric not in found_keys:
            raise ValueError(f"`{metric}` was not found in Tensorboard records. Found keys = `{found_keys}`")

    return path, data, logs


def extract_global_attributes_and_join_into_logs(opt, metrics, data, logs):
    # args merging
    if opt.merge_args != "":
        data = merge_args(data, opt.merge_args)

    # drop columns that contain the same attributes (they will be dropped from `data`)
    nunique = data.apply(pd.Series.nunique)
    global_attributes_key = [k for k in list(nunique[nunique == 1].index) if k not in ['id']]

    # remove exceptions from the `global_attributes` if they have to be indexed from `data` later on
    for k in metrics['pivot_metrics'] + ['seed', 'dataset'] + metrics['keys']:  # `seed` and `dataset` are never dropped
        if k in global_attributes_key:
            global_attributes_key.remove(k)

    # store the global attributes
    global_attributes = {k: data[k].values[0] for k in global_attributes_key}

    # finally drop the global attributes from `data`
    data.drop(global_attributes_key, axis=1, inplace=True)

    # join the columns of `data` that are not `metrics` (so iw, dataset, ...) into logs
    _keys_to_merge = [k for k in data.keys() if k not in metrics['pivot_metrics']]
    logs = logs.merge(data[_keys_to_merge], left_on="id", right_on="id", how='left')

    return data, logs, global_attributes


def aggregate_metrics(data, logs, metrics):
    # aggregate `metrics` from logs and join into `df` (i.e. compute `max. L_K` and join into `data`)
    for m, agg_fn in zip(metrics['pivot_metrics'], metrics['pivot_metrics_agg_fns']):
        logs_m = logs[logs["_key"] == m]
        agg_log_m = pd.DataFrame(logs_m[['id', "_value"]].groupby('id')["_value"].apply(agg_fn))
        agg_log_m.rename(columns={'_value': m}, inplace=True)
        data = data.merge(agg_log_m, left_on="id", right_on="id", how='right')

    # sort by the first metric
    data = data.sort_values(metrics['pivot_metrics'][0], ascending=False)

    return data


def parse_keys_headers_metrics(opt):
    """parse the keys, headers and metrics"""

    # define the `keys` that used to slice DataFrames and color/style plots (i.e. `iw`, `alpha`, etc..)
    keys = list(opt.keys.replace(" ", "").split(','))

    # define the basic curves plots
    curves_metrics = opt.metrics.replace(" ", "").split(',')

    # define the metrics for the `detailed` plots
    detailed_metrics = list(opt.detailed_metrics.replace(" ", "").split(','))
    if detailed_metrics[0] == "":
        detailed_metrics = []

    # define the `pivot table` metrics (aggregated statistics)
    pivot_metrics_agg_ids, pivot_metrics = zip(
        *[parse_pivot_metric(u) for u in opt.pivot_metrics.replace(" ", "").split(",")])
    pivot_metrics = list(pivot_metrics)
    pivot_metrics_agg_fns = map(get_agg_fn, pivot_metrics_agg_ids)
    pivot_metrics_agg_ids = {m: f for m, f in zip(pivot_metrics, pivot_metrics_agg_ids)}

    # gather all the metrics to be read from the tensorboard logs
    all_metrics = list(set(curves_metrics + detailed_metrics + pivot_metrics))
    _headers, _all_metrics = zip(*[u.split(":") for u in all_metrics])

    # gather all metrics
    all_metrics_by_header = defaultdict(list)
    [all_metrics_by_header[h].append(k) for (h, k) in zip(_headers, _all_metrics)]

    return {
        'keys': keys,
        'curves_metrics': curves_metrics,
        'detailed_metrics': detailed_metrics,
        'pivot_metrics': pivot_metrics,
        'pivot_metrics_agg_fns': pivot_metrics_agg_fns,
        'pivot_metrics_agg_ids': pivot_metrics_agg_ids,
        'all_metrics': all_metrics,
        'all_metrics_by_header': all_metrics_by_header
    }


def build_pivot_table(opt, data, metrics):
    """average over seed and attributes not refereced in `keys`"""

    def aggfunc(serie):
        """function use to aggregate the results into the pivot table"""
        mean = np.mean(serie)
        std = np.std(serie)
        return f"{mean:{opt.float_format}} +/- {std:{opt.float_format}} (n={len(serie)})"

    # excludes `keys` and `all_metrics`
    _columns = []
    _excluded_keys = metrics['all_metrics'] + _columns + ["seed"] + metrics['keys'][::-1]
    _index = [k for k in data.keys() if k not in _excluded_keys]
    _index = _index + metrics['keys'][::-1]

    # pivot table
    pivot = data.pivot_table(index=_index, columns=_columns, values=metrics['pivot_metrics'], aggfunc=aggfunc)

    # build another pivot table using only the first metric to sort the original pivot table accordingly
    mean_pivot = data.pivot_table(index=_index, values=metrics['pivot_metrics'][0], aggfunc=np.mean)
    mean_pivot = mean_pivot.sort_values(metrics['pivot_metrics'][0], ascending=False)

    # sort the original pivot table
    pivot = pivot.reindex(mean_pivot.index)

    # sort index
    for idx in _index[::-1][1:]:
        pivot.sort_index(level=idx, sort_remaining=False, inplace=True)

    return pivot


def exponential_moving_average(logs, value=0.5):
    """
    Curve smoothing using Exponential Moving Average
    """
    logs.sort_values("step", inplace=True)
    sort_index = [k for k in logs.keys() if k not in ['step', '_value', '_key']] + ['_key', 'step']

    logs.set_index(keys=sort_index, inplace=True)
    for idx in sort_index[::-1]:
        logs.sort_index(level=idx, sort_remaining=False, inplace=True)

    for k, (idx, record) in enumerate(logs.groupby(level=list(range(len(sort_index) - 1)))):
        record = record['_value']

        # exponential moving average
        record = record.ewm(alpha=1 - value).mean()

        logs.loc[idx, :] = record

    logs.reset_index(inplace=True)

    return logs


def downsample(logs, nsamples, logger=None):
    """downsample logs to speed-up plotting"""

    _last_indexes = ['seed', '_key', 'step']
    _index = [k for k in logs.keys() if k != '_value' and k not in _last_indexes]
    _index += _last_indexes

    # get max step
    M = logs['step'].max()
    bins = [int(s) for s in range(0, M, M // nsamples)]

    n_full = len(logs['step'].unique())
    ratio = nsamples / n_full

    if logger is not None:
        logger.info(f"Downsampling data.. (n={nsamples}, ratio = {ratio})")
    logs.reset_index(level=-1, inplace=True)
    _index.remove("step")
    logs = logs.groupby(_index + [pd.cut(logs.step, bins)]).mean()
    logs.index.rename(level=[-1], names=['step_bucket'], inplace=True)
    logs.reset_index(inplace=True)
    logs.drop("step_bucket", 1, inplace=True)

    return logs


def parse_ylims(opt):
    if len(opt.ylims):
        ylims = [(u.split(":")[:-2], u.split(":")[-2:]) for u in opt.ylims.replace(" ", "").split(',')]
        return {":".join(u[0]): [eval(u[1][0]), eval(u[1][1])] for u in ylims}
    else:
        return {}
