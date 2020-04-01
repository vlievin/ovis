import argparse
import json
import os
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotmap import DotMap
from tqdm import tqdm

sns.set()


def plot_logs(logs, path, metrics, main_key, style_key=None, ylims=dict()):
    """
    Make a grid of line plots with std intervals. Each subplot correspond to one metric.
    :param path: output file
    :param logs: dataframe containing all the training data per time step
    :param metrics: metrics to plot (elbo, nll, ...)
    :param main_key: key to used for coloring
    :param style_key: key used for styling (Optional)
    :param ylims: dictionary for the y axis boundaries (Optional)
    :return: None
    """

    N = len(metrics)
    ncols = 2
    nrows = N // ncols

    if N > ncols * nrows:
        nrows += 1
        metrics = list(metrics) + [metrics[-1] for _ in
                                   range((ncols * nrows - len(metrics)))]  # repeat the last plot for the legend

    hue_order = list(logs[main_key].unique())
    step_min = np.percentile(logs['step'].values.tolist(), 10)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    for i, k in tqdm(list(enumerate(metrics)), desc="|    subplots"):
        u = i // ncols
        v = i % ncols
        ax = axes[u, v]

        sns.lineplot(x="step", y="_value",
                     hue=main_key,
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

        if i < len(metrics) - 1:
            ax.get_legend().remove()

    # draw legend in the last plot
    # patches = [mpatches.Patch(color=sns.color_palette()[i], label=key) for i, key in enumerate(hue_order)]
    # axes[nrows - 1, ncols - 1].legend(handles=patches)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def spot_on_plot(logs, path, metrics, main_key, auxiliary_key, style_key=None, ylims=dict()):
    """make a grid of plot metrics x aux. keys values"""
    aux_keys = logs[auxiliary_key].unique()
    nrows = len(metrics)
    ncols = len(aux_keys)

    if nrows == 0 or ncols == 0:
        return None

    hue_order = list(logs[main_key].unique())
    step_min = np.percentile(logs['step'].values.tolist(), 10)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

    for j, aux_key in tqdm(list(enumerate(sorted(aux_keys))), desc="|  aux. keys"):
        aux_data = logs[logs[auxiliary_key] == aux_key]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j]
                data = aux_data[aux_data["_key"] == metric]

                sns.lineplot(x="step", y="_value",
                             hue=main_key,
                             hue_order=hue_order,
                             style=style_key,
                             data=data, ax=ax,
                             # palette=sns.color_palette("mako_r", 4)
                             )

                if i == 0:
                    ax.set_title(f"{auxiliary_key} = {aux_key}")
                ax.set_ylabel(metric)
                # y lims
                if metric in ylims:
                    ax.set_ylim(ylims[metric])
                else:
                    ys = data[data['step'] > step_min]['_value'].values.tolist()
                    if len(ys):
                        a, b = np.percentile(ys, [25, 75])
                        M = b - a
                        k = 1.5
                        ax.set_ylim([a - k * M, b + k * M])

                if  not (i == len(metrics) - 1 and  j == len(aux_keys) - 1):
                    ax.get_legend().remove()
            except:
                warnings.warn(f">> spot-on plot: couldn't generate the axis `ax[{i}, {j}]`")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pivot_plot(df, path, metrics, main_key, auxiliary_key, style_key=None, ylims=dict()):
    """make grid of pointplot [metric x dataset], where each point plot is [aux_key vs. metric] (e.g. iw vs. avg log_snr) """

    color_palette = sns.color_palette()
    line_styles = ["-", "--", ":", "-."]
    markers = ["x", "+", "v", "^", "1", "2", "*", "+"]
    dsets = df['dataset'].unique()
    ncols = len(dsets)
    nrows = len(metrics)

    if style_key is not None:
        main_keys =list(df[main_key].unique())
        style_keys = list(df[style_key].unique())
        key_name = f"{main_key}-{style_key}"
        df[key_name] = [f"{x}-{y}" for (x,y) in zip(df[main_key].values, df[style_key].values)]
        df = df.drop(main_key, 1)
        df = df.drop(style_key, 1)

        hue_order, linestyles, palette = [], [], []
        for y, _linestyle in zip(style_keys, line_styles):
            for x, hue in zip(main_keys, color_palette):
                hue_order += [f"{x}-{y}"]
                linestyles += [_linestyle]
                palette += [hue]


    else:
        key_name = main_key
        hue_order = list(df[main_key].unique())
        linestyles = [line_styles[0] for _ in hue_order]
        palette = color_palette


    if nrows == 0 or ncols == 0:
        return None

    hue_order = list(df[key_name].unique())
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

    for j, dset in enumerate(dsets):
        dset_data = df[df["dataset"] == dset]

        for i, metric in enumerate(metrics):

            try:
                ax = axes[i, j]

                sns.pointplot(x=auxiliary_key, y=metric, hue=key_name, data=dset_data, ax=ax, hue_order=hue_order, linestyles=linestyles, color_palette=palette, markers=markers, capsize=.2)
                plt.setp(ax.lines, alpha=.7)

                if i == 0:
                    ax.set_title(f"Dataset = {dset}")
                ax.set_ylabel(metric)

                if  not (i == len(metrics) - 1 and j == len(dsets) - 1):
                    ax.get_legend().remove()
            except:
                warnings.warn(f">> pivot plot: couldn't generate the axis `ax[{i}, {j}]`")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()