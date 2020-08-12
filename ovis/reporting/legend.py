import warnings
from collections import Counter, defaultdict
from copy import copy

import matplotlib
from matplotlib import pyplot as plt

from ovis.reporting.style import ESTIMATOR_ORDER, ESTIMATOR_GROUPS, ESTIMATOR_DISPLAY_NAME


class Legend():
    """A legend handler for subplots (draws a single legend on top of the subplots)"""

    def __init__(self, figure):
        self.figure = figure
        self.legend_infos = []

    def update(self, ax):
        """get legend info from the current axis"""
        self.legend_infos += list(zip(*ax.get_legend_handles_labels()))

    def update_from_infos(self, infos):
        """get legend info from the current axis"""
        self.legend_infos += list(infos)

    def reset_linestyles(self):
        self.legend_infos = [(copy(handle), label) for handle, label in self.legend_infos]
        [handle.set_linestyle("-") for handle, label in self.legend_infos]

    def draw(self, group=False, alpha=1, **kwargs):
        legend_infos = {label: handle for handle, label in self.legend_infos}
        if len(legend_infos) == 0:
            return

        labels, handles = zip(*legend_infos.items())

        # legend for linestyles
        style_labels, style_handles = [], []

        # special case (when using seaborn)
        if 'estimator' == labels[0]:
            labels, handles = labels[1:], handles[1:]

        # parse `style` labels (when using seaborn)
        for special_key in ['iw', 'warmup', 'alpha_init', 'alpha']:
            if special_key in labels:
                q = labels.index(special_key)
                style_labels, style_handles = labels[q + 1:], handles[q + 1:]
                labels, handles = labels[:q], handles[:q]

                if special_key == 'iw':
                    # update iw label name "x" -> "K=x"
                    style_labels = [f"K={l}" for l in style_labels]
                if special_key == 'warmup':
                    _f = lambda l: eval(l) if isinstance(l, str) else l
                    style_labels = [r"$\operatorname{with}\ \alpha\ \operatorname{warmup}$" if _f(l) > 0 else r"$\operatorname{without}\ \alpha\ \operatorname{warmup}$" for l in style_labels]
                if special_key == 'alpha_init':
                    style_labels = [r"$\alpha_{\operatorname{init}}=" + f"{l}$" for l in style_labels]
                if special_key == 'alpha':
                    style_labels = [r"$\alpha=" + f"{l}$" for l in style_labels]

        # sort labels
        if all([l in ESTIMATOR_ORDER for l in labels]):
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: ESTIMATOR_ORDER.index(x[0])))
            groups = [ESTIMATOR_GROUPS[l] for l in labels]
        else:
            groups = [0 for _ in labels]
            warnings.warn(f"Not all labels = `{labels}` are in `ESTIMATOR_ORDER`, the legend won't be sorted")

        # group number for the styles
        style_groups = [max(groups) + 1 for l in style_labels]

        # format names
        labels = [ESTIMATOR_DISPLAY_NAME.get(l, l) for l in labels]

        # define number of columns
        all_groups = groups + style_groups

        # grouping
        if group and len(set(all_groups)) > 1:
            ncol = len(set(all_groups))

            # get longer group
            most_common_key, max_length = Counter(all_groups).most_common(1)[0]
            _all_labels = list(labels) + list(style_labels)
            _all_handles = list(handles) + list(style_handles)

            # build dicitonaruy {group : [labels]}
            all_labels, all_handles = defaultdict(list), defaultdict(list)
            for g, l, h in zip(all_groups, _all_labels, _all_handles):
                all_labels[g] += [l]
                all_handles[g] += [h]

            # extend all groups to match max length
            _null_patch = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                                                       visible=False)
            for g in all_groups:
                all_labels[g] = all_labels[g] + (max_length - len(all_labels[g])) * [""]
                all_handles[g] = all_handles[g] + (max_length - len(all_handles[g])) * [_null_patch]

            # flatten everything
            all_labels = [l for ls in all_labels.values() for l in ls]
            all_handles = [l for ls in all_handles.values() for l in ls]

            # number of rows
            nrow = max_length

        else:
            # append style legend
            all_labels = list(labels) + list(style_labels)
            all_handles = list(handles) + list(style_handles)
            ncol = len(labels)
            nrow = 1

        # infer sizes
        width, height = self.figure.get_size_inches()
        legend_base_size = 0.5
        legend_row_size = 0.3
        margin = 0.5
        legend_height = legend_base_size + nrow * legend_row_size

        # set tight layout and add margin at the top for the legend
        plt.tight_layout()
        self.figure.subplots_adjust(top=1 - (legend_height + margin) / height)

        # draw the legend
        legend = self.figure.legend(handles=all_handles,
                                    labels=all_labels,
                                    ncol=ncol,
                                    loc='lower center',
                                    bbox_to_anchor=(0, 1 - (legend_height) / height, 1, 1),
                                    fancybox=False,
                                    shadow=False,
                                    fontsize=kwargs.pop('fontsize', 'medium' if ncol < 8 else 'x-small'),
                                    **kwargs)

        if alpha is not None:
            for l in legend.get_lines():
                l.set_alpha(1)