import argparse
import os
from shutil import rmtree

from booster.utils import logging_sep

from ovis.reporting.parsing import read_experiments, \
    extract_global_attributes_and_join_into_logs, aggregate_metrics, parse_keys_headers_metrics, build_pivot_table, \
    exponential_moving_average, downsample, parse_ylims
from ovis.reporting.plotting import pivot_plot, basic_curves_plot, detailed_curves_plot
from ovis.reporting.style import *
from ovis.reporting.style import LOG_PLOT_RULES, METRIC_DISPLAY_NAME, format_estimator_name
from ovis.training.logging import get_loggers


def report():
    """
    Read a entire `experiment` folder (i.e. containing multiple runs),
    aggregate the data, dump .csv files and generate the plots.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/', help='experiment directory')
    parser.add_argument('--output', default='reports/', help='output directory')
    parser.add_argument('--exp', default='mini-vae', type=str, help='experiment id')
    parser.add_argument('--include', default='', type=str, help='filter run by id')
    parser.add_argument('--exclude', default='', type=str, help='filter run by id')
    parser.add_argument('--pivot_metrics', default='max:train:loss/L_k,min:train:loss/kl_q_p,mean:train:grads/snr',
                        type=str,
                        help='comma separated list of metrics to report in the table, '
                             'the prefix defines the aggregation function (min, mean, max)')
    parser.add_argument('--metrics',
                        default='train:loss/L_k,train:loss/kl_q_p,train:loss/ess,train:grads/snr',
                        type=str,
                        help='comma separated list of keys to read from the tensorboard logs')
    parser.add_argument('--ylims', default='', type=str,
                        help='comma separated list of limit values for the curve plot (syntax: key:min:max), '
                             'example: `train:loss/ess:1:3.5,train:loss/L_k:-89:-84,test:loss/L_k:-91:-87`')
    parser.add_argument('--keys', default='dataset, estimator, iw', type=str,
                        help='comma separated list of keys to include in the report by decreasing order of importance, '
                             'more than 4 keys is not yet handled for plotting.')
    parser.add_argument('--detailed_metrics', default='train:loss/L_k,train:loss/kl_q_p,train:loss/ess,train:grads/snr',
                        type=str, help='comma separated list of keys for the `detailed` plots. ')
    parser.add_argument('--parse_estimator_args', action='store_true',
                        help='parse estimator arguments such that `ovis-gamma1` -> `ovis` + gamma=1')
    parser.add_argument('--latex', action='store_true', help='print as latex table')
    parser.add_argument('--float_format', default=".2f", help='float format')
    parser.add_argument('--downsample', default=0, type=int,
                        help='maximum number of point for the curves (downsampling)')
    parser.add_argument('--ema', default=0, type=float, help='exponential moving average')
    parser.add_argument('--non_completed', action='store_true', help='allows reading non-completed runs')
    parser.add_argument('--max_records', default=-1, type=int,
                        help='only read the first `max_records` data point (`-1` = no limit)')
    parser.add_argument('--merge_args', default='', type=str, help='comma separated list of args to be merged')
    opt = parser.parse_args()

    # define the run identifier
    run_id = opt.exp
    if len(opt.include):
        run_id += f"-inc={opt.include}"
    if len(opt.exclude):
        run_id += f"-exc={opt.exclude}"
    if opt.ema > 0:
        run_id += f"-ema{opt.ema}"

    # define, create output directory and get the logger
    output_path = os.path.join(opt.output, run_id)
    if os.path.exists(output_path):
        rmtree(output_path)
    os.makedirs(output_path)
    logger, *_ = get_loggers(output_path, keys=['report'], format="%(message)s")

    def print_df(df):
        """custom print function based on `opt.latex` and `opt.float_format`"""
        if opt.latex:
            logger.info(df.to_latex(float_format=f"%{opt.float_format}"))
        else:
            logger.info(df)

    """
    read experiments, parse metrics and join attributes
    """

    metrics = parse_keys_headers_metrics(opt)
    path, data, logs = read_experiments(opt, metrics, logger)
    data, logs, global_attributes = extract_global_attributes_and_join_into_logs(opt, metrics, data, logs)
    data = aggregate_metrics(data, logs, metrics)
    # drop id (exp_id used for joins)
    data.drop('id', 1, inplace=True)
    logs.drop('id', 1, inplace=True)
    # format estimator names
    data['estimator'] = list(map(format_estimator_name, data['estimator'].values))
    logs['estimator'] = list(map(format_estimator_name, logs['estimator'].values))

    """
    print all results
    """

    logger.info(f"{logging_sep('=')}\nGlobal Attributes\n{logging_sep('-')}")
    for k, v in global_attributes.items():
        logger.info(f"{k} : {v}")
    logger.info(f"{logging_sep('-')}\nExperiment path : {os.path.abspath(path)}\n{logging_sep('-')}")
    logger.info(f"Varying Parameters: {[k for k in data.keys() if k not in metrics['pivot_metrics']]}")
    logger.info(f"{logging_sep('-')}\nData (sorted by {metrics['pivot_metrics'][0]})\n{logging_sep('-')}")
    print_df(data)
    logger.info(logging_sep("="))

    """
    build the pivot table and print
    """

    pivot = build_pivot_table(opt, data, metrics)
    logger.info("\n" + logging_sep('='))
    logger.info("Pivot table\n" + logging_sep())
    print_df(pivot)
    logger.info(logging_sep('='))

    # save to file
    data.to_csv(os.path.join(output_path, "pivot.csv"))

    """
    post processsing: smoothing + downsampling
    """

    if opt.ema > 0:
        logs = exponential_moving_average(logs, opt.ema)

    if opt.downsample > 0:
        logs = downsample(logs, opt.downsample, logger=logger)

    """drop nans + save to file"""
    logs.dropna(inplace=True)
    logs.to_csv(os.path.join(output_path, "curves.csv"))

    """plotting pivot plots, curve plots and detailed plots"""
    # set style
    set_matplotlib_style()

    # plot for all auxiliary keys
    ylims = parse_ylims(opt)

    # ensure the first key to be `dataset`
    keys = metrics['keys']
    if logs['dataset'].nunique() > 1 and keys[0] != 'dataset':
        keys = ['dataset'] + keys

    # define keys used for styling
    cat_key = keys[0]
    main_key = keys[1]  # color
    aux_key = keys[2] if len(keys) > 2 else None  # line style (in main plot)
    third_key = keys[3] if len(keys) > 3 else None  # line style (in auxiliary plots)

    meta = {'log_rules': LOG_PLOT_RULES, 'metric_dict': METRIC_DISPLAY_NAME,
            'agg_fns': metrics['pivot_metrics_agg_ids']}

    if logs[cat_key].nunique() > 1:
        level = 1
        # pivot plot
        logger.info(f"|- Generating pivot plots for all keys {cat_key} ..")
        _path = os.path.join(output_path, f"{level}-pivot-plot-{cat_key}-hue={main_key}.png")
        pivot_plot(data, _path, metrics['pivot_metrics'], cat_key, main_key, aux_key, style_key=third_key, **meta)

    # plot all data for each key
    for cat in logs[cat_key].unique():
        level = 2
        print(logging_sep())
        logger.info(f"[{cat_key} = {cat}]")
        # slice data
        cat_logs = logs[logs[cat_key] == cat]
        cat_data = data[data[cat_key] == cat]

        logger.info(f"|- Generating pivot plots with key = {cat_key} ..")
        _path = os.path.join(output_path, f"{level}-{cat_key}={cat}-pivot-plot-hue={main_key}-style={third_key}.png")
        pivot_plot(cat_data, _path, metrics['pivot_metrics'], cat_key, main_key, aux_key, style_key=third_key, **meta)

        logger.info(f"|- Generating simple plots for all aux_key = {aux_key}")
        _path = os.path.join(output_path, f"{level}-{cat_key}={cat}-curves-hue={main_key}-style={aux_key}.png")
        basic_curves_plot(cat_logs, _path, metrics['curves_metrics'], main_key, ylims=ylims, style_key=aux_key, **meta)

        if aux_key is not None:
            level = 3

            # detailed plots
            if len(metrics['detailed_metrics']):
                logger.info(f"|- Generating detailed plots for aux. key = {aux_key}")
                _path = os.path.join(output_path,
                                     f"{level}{cat_key}={cat}-detailed-plot-hue={main_key}-style={aux_key}.png")
                detailed_curves_plot(cat_logs, _path, metrics['detailed_metrics'], main_key, aux_key,
                                     style_key=third_key, ylims=ylims, **meta)

            # on plot for each auxiliary key
            # for each aux key..
            aux_key_values = list(cat_logs[aux_key].unique())
            for i, v in enumerate(sorted(aux_key_values)):
                aux_cat_logs = cat_logs[cat_logs[aux_key] == v]
                logger.info(f"|--- Generating simple plots for {aux_key} = {v} [{i + 1} / {len(aux_key_values)}]")

                # auxiliary plot
                _path = os.path.join(output_path,
                                     f"{level}-{cat_key}={cat}-{aux_key}={v}-curves-hue={main_key}-style={third_key}.png")
                basic_curves_plot(aux_cat_logs, _path, metrics['curves_metrics'], main_key, ylims=ylims,
                                  style_key=third_key, **meta)


if __name__ == '__main__':
    report()
