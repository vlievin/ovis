import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ovis.reporting.style import *
from ovis.reporting.style import set_matplotlib_style
from ovis.reporting.utils import smooth, update_labels, lighten
from ovis.utils.utils import Header

parser = argparse.ArgumentParser()
parser.add_argument('--figure', default='left', help='[left, right]')
parser.add_argument('--root', default='reports/', help='experiment directory')
parser.add_argument('--exp', default='', type=str,
                    help='experiment id [default use the exp name specified in the Readme.md]')

# keys
parser.add_argument('--style_key', default='iw', help='style key')
parser.add_argument('--metric', default='train:loss/L_k', help='metric to display')

# plot config
parser.add_argument('--desaturate', default=0.9, type=float, help='desaturate hue')
parser.add_argument('--lighten', default=1.2, type=float, help='lighten hue')
parser.add_argument('--alpha', default=0.8, type=float, help='opacity')
parser.add_argument('--linewidth', default=1.2, type=float, help='line width')
opt = parser.parse_args()

# matplotlibg style
set_matplotlib_style()
plot_style = {
    'linewidth': opt.linewidth,
    'alpha': opt.alpha
}

# experiment directory
default_exps = {'left': 'sigmoid-belief-network-inc=iwbound', 'right': 'sigmoid-belief-network-inc=iwrbound'}
if opt.exp == '':
    root = os.path.join(opt.root, default_exps[opt.figure])
else:
    root = os.path.join(opt.root, opt.ex)

# read data
data = pd.read_csv(os.path.join(root, 'curves.csv'))
filtered_data = data
filtered_data = filtered_data[filtered_data['_key'] == opt.metric]

print(data)

# plot the figure
figure = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)
ax = plt.gca()

# color
hue_order = list(filtered_data['estimator'].unique())
palette = [ESTIMATOR_STYLE[h_key]['color'] for h_key in hue_order]
palette = [sns.desaturate(c, opt.desaturate) for c in palette]
palette = [lighten(c, opt.lighten) for c in palette]

# linestyles & markers
style_order = list(sorted(filtered_data[opt.style_key].unique()))
line_styles = [":", "--", "-"]
markers = ["x", "^", "o"]

# draw
with Header("Records"):
    for e, estimator in enumerate(hue_order):
        for s, style in enumerate(style_order):
            sub_df = filtered_data[(filtered_data['estimator'] == estimator) & (filtered_data[opt.style_key] == style)]
            sub_df = sub_df.groupby('step')['_value'].mean()
            x = sub_df.index.values
            y = sub_df.values
            color = palette[hue_order.index(estimator)]

            if len(y):
                y = smooth(y, window_len=15)
                plt.plot(x, y, color=color, linestyle=line_styles[s], **plot_style)

                idx = np.round(np.linspace(0, len(x) - 1, 6)).astype(int)

                print(f"{estimator} - {opt.style_key} = {style} :  max. {opt.metric} = {max(y):.3f}")

                marker = markers[s]
                # marker = ESTIMATOR_STYLE[estimator]['marker']
                plt.plot(x[idx], y[idx], color=color, linestyle="", marker=marker, markersize=5, alpha=0.9)

# set axis labels
ax.set_ylabel(opt.metric)
update_labels(np.array(ax), METRIC_DISPLAY_NAME, agg_fns=dict())

if opt.figure == 'left':
    infos = [(Line2D([0], [0], color="black", linewidth=3, linestyle=''), r"$\operatorname{OVIS}_{\operatorname{MC}}$")]
    for estimator, name in zip(["ovis-S50", "ovis-S10"], [r"$S=50, \alpha=0$", r"$S=10, \alpha=0$"]):
        color = ESTIMATOR_STYLE[estimator]['color']
        patch = Line2D([0], [0], color=color, linewidth=3, linestyle='-')
        label = name
        infos += [(patch, label)]

    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"$\operatorname{OVIS}_{\boldsymbol{\sim}}$")]
    for estimator, name in zip(["ovis-gamma0", "ovis-gamma1"], [r"$\gamma=0, \alpha=0$", r"$\gamma=1, \alpha=0$"]):
        color = ESTIMATOR_STYLE[estimator]['color']
        patch = Line2D([0], [0], color=color, linewidth=3, linestyle='-')
        label = name
        infos += [(patch, label)]

    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"\textit{Baseline}")]
    for estimator, name in zip(["vimco-arithmetic"], [None]):
        color = ESTIMATOR_STYLE[estimator]['color']
        patch = Line2D([0], [0], color=color, linewidth=3, linestyle='-')
        label = ESTIMATOR_DISPLAY_NAME[estimator]
        infos += [(patch, label)]

    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), " ")]
    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"\textit{Particles}")]
    for s, style in enumerate(style_order):
        patch = Line2D([0], [0], color="#666666", marker=markers[s], linestyle=line_styles[s], markersize=5)
        label = f"K = {style}"
        infos += [(patch, label)]
elif opt.figure == 'right':
    infos = [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"$\operatorname{OVIS}_{\operatorname{MC}}$")]
    for estimator, name in zip(["ovis-S50", "ovis-S10", "ovis-S1"],
                               [r"$S=50, \alpha \geq 0$", r"$S=10, \alpha \geq 0$", r"$S=1, \alpha=\geq 0$"]):
        color = ESTIMATOR_STYLE[estimator]['color']
        patch = Line2D([0], [0], color=color, linewidth=3, linestyle='-')
        label = name
        infos += [(patch, label)]

    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"$\operatorname{OVIS}_{\boldsymbol{\sim}}$")]
    for estimator, name in zip(["ovis-gamma1"], [r"$\gamma=1, \alpha \geq 0$"]):
        color = ESTIMATOR_STYLE[estimator]['color']
        patch = Line2D([0], [0], color=color, linewidth=3, linestyle='-')
        label = name
        infos += [(patch, label)]

    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"\textit{Baseline}")]
    for estimator, name in zip(["tvo"], [None]):
        color = ESTIMATOR_STYLE[estimator]['color']
        patch = Line2D([0], [0], color=color, linewidth=3, linestyle='-')
        label = ESTIMATOR_DISPLAY_NAME[estimator]
        infos += [(patch, label)]

    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), " ")]
    infos += [(Line2D([0], [0], color=color, linewidth=3, linestyle=''), r"\textit{Particles}")]
    for s, style in enumerate(style_order):
        patch = Line2D([0], [0], color="#666666", marker=markers[s], linestyle=line_styles[s], markersize=5)
        label = f"K = {style}"
        infos += [(patch, label)]
else:
    raise ValueError(f"Unknown `--figure` argument = `{opt.figure}`. Accepted values: left, right")

# shrink plot to leave room for the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
lines, labels = zip(*infos)
ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small', fancybox=False, shadow=False)
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(STEP_FORMAT))
loc = matplotlib.ticker.MultipleLocator(base=1.0)
ax.yaxis.set_major_locator(loc)

plt.ylim([-92.5, -86.2])
plt.savefig(os.path.join(root, f'figure3_{opt.figure}.png'), bbox_inches='tight')
plt.close()


print(f"Output = {os.path.abspath(root)}")