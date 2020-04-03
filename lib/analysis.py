import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import imageio

_key = "|estimate - score|"
k_out = 5


def analyse_control_variate(x, model, c_configs, c_estimators, c_names, writer_train, seed=None, global_step=None):
    data = []

    for c_onf, c_est, c_name in zip(c_configs, c_estimators, c_names):

        print("- analysis:", c_name)

        if seed is not None:
            _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(seed)

        bs, mc, iw = x.size(0), c_est.mc, c_est.iw
        loss, diagnostics, output = c_est(model, x, **c_onf)

        if "L_k" in output.keys() and "v" in output.keys():

            # get score
            L_k = output["L_k"][:, :, None].expand(bs, mc, iw)
            v = output["v"].expand(bs, mc, iw)
            dqlogits = output["dqlogits"].view(bs, mc, iw, -1)
            dqlogits_hat = dqlogits.mean(dim=(0, 1, 2), keepdim=True)
            dqlogits_prod = (dqlogits * dqlogits_hat).sum(-1)

            # get estimates
            L_hat = output.get("L_hat", torch.zeros_like(L_k)[..., None]).mean(-1)  # avg. over Nz
            v_hat = output.get("v_hat", torch.zeros_like(v)[..., None]).mean(-1)

            # print(f">> {c_name}: L_k = {L_k.shape}, L_hat = {output['L_hat'].shape} | {L_hat.shape}, v = {v.shape}, v_hat = {v_hat.shape}")

            keys = ["a", "b", "c", "d"]
            labels = [r"$ \hat{h} h_k^T (Z_{1:K} - v_k)  $", r"$Z_{1:K} - v_k$", r"$Z_{1:K}$", r"$v_k$"]
            scores = [dqlogits_prod * (L_k - v), L_k - v, L_k, v]
            estimates = [dqlogits_prod * (L_hat - v_hat), L_hat - v_hat, L_hat, v_hat]

            for i, (key, label, score, estimate) in enumerate(zip(keys, labels, scores, estimates)):

                # print(f"{c_name}: {label}: score = {score.shape}, estimate = {estimate.shape}")
                score = score.contiguous().view(-1).data.cpu().numpy().tolist()
                estimate = estimate.contiguous().view(-1).data.cpu().numpy().tolist()
                assert len(score) == len(estimate), f"{c_name}: score = {len(score)}, estimate = {len(estimate)}"
                for seed, (s, e) in enumerate(zip(score, estimate)):
                    data += [{'estimator': c_name, 'label': label, 'score': s, _key: np.abs(e - s), 'key':key, 'seed': seed}]

    if len(data):
        data = pd.DataFrame(data)

        labels = data['label'].unique()
        ncols = len(labels)
        nrows = 2
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4 * ncols, 3 * nrows))
        for i, label in enumerate(labels):
            ax = axes[0, i]

            _data = data[data["label"] == label]

            sns.scatterplot("score", _key, data=_data, hue="estimator", ax=ax, alpha=0.3, s=15)

            ax.set_ylabel(r"$| \hat{y} - \hat{y}^{-k} |$")
            ax.set_xlabel(r"$\hat{y} $")
            ax.set_title(r"$\hat{y} = $" + label)

            # axis scaling
            if any(["covbaseline" in n for n in c_names]):
                _names = [n for n in c_names if "covbaseline" in n]
                _data[_data["estimator"].isin(_names)]

            ys = _data[_key].values.tolist()
            if label != r"$v_k$":
                a, b = np.percentile(ys, [25, 75])
                y_b = b + k_out * (b - a)
                y_a = a - k_out * (b - a)
                ax.set_ylim([y_a, y_b])

        for i, label in enumerate(labels):

            _data = data[data["label"] == label]

            # filter outliers
            # values = _data[_key].values
            # a, b = np.percentile(values, [25, 75])
            # y_b = b + k_out * (b - a)
            # y_a = a - k_out * (b - a)
            # _data = _data[(_data[_key] <= y_b) & (_data[_key] >= y_a)]

            for c_name in c_names:
                c_data = _data[_data["estimator"] == c_name]

                values = c_data[_key].values

                ax = axes[1, i]
                sns.distplot(values, ax=ax, label=c_name, rug=True, kde=False, bins=100)

            # ax = axes[2, i]
            # sns.swarmplot(x="estimator", y=_key, data=_data, palette="Set2", ax=ax, alpha=0.5)

            axes[1, i].set_xlabel(r"$| \hat{y} - \hat{y}^{-k} |$")
            # axes[2, i].set_xlabel(r"$| \hat{y} - \hat{y}^{-k} |$")

            # ys = _data[_key].values.tolist()
            # if label != r"$v_k$":
            #     a, b = np.percentile(ys, [25, 99.9999])
            #     ax.set_xlim([0, b])
            #
            # u = 1.5 * 1./(b-a)
            # ax.set_ylim([- 1e-1 * u , u])

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        # plt.close()

        # draw canvas to numpy array
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        # log to tensorboard
        writer_train.add_image("analysis/control_variates", image.transpose([2, 0, 1]), global_step)


        # pair plot
        # _labels = ["h_x_score", "score"]
        # for i, key in enumerate(keys[:2]):
        #     _data = data[data["key"] == key].pivot(index="seed", columns="estimator", values=_key)
        #
        #     g = sns.pairplot(data=_data,
        #              plot_kws=dict(s=10, alpha=0.2))
        #
        #     _path = '.tmp_plot.png'
        #     plt.savefig(_path)
        #     plt.close()
        #     image = imageio.imread(_path)
        #     writer_train.add_image(f"analysis/pairplot_{_labels[i]}", image.transpose([2, 0, 1]), global_step)







        if seed is not None:
            torch.manual_seed(_seed)

        model.zero_grad()
