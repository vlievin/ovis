import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import imageio

# sns.set()

_key = "|estimate - score|"
k_out = 1.5
colors = sns.color_palette()


def analyse_control_variate_individual(x, model, c_configs, c_estimators, c_names, writer_train, seed=None, global_step=None):
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

            # get estimates
            L_hat = output.get("L_hat", torch.zeros_like(L_k)[..., None])
            v_hat = output.get("v_hat", torch.zeros_like(v)[..., None])

            dqlogits_hat = (dqlogits * (L_k[..., None] - v[..., None] - L_hat - v_hat)).mean(dim=(0, 1), keepdim=True)
            dqlogits_prod = (dqlogits * dqlogits_hat).sum(-1)

            L_hat = L_hat.mean(-1)
            v_hat = v_hat.mean(-1)

            # print(f">> {c_name}: L_k = {L_k.shape}, L_hat = {output['L_hat'].shape} | {L_hat.shape}, v = {v.shape}, v_hat = {v_hat.shape}")

            keys = ["a", "b", "c", "d"]
            labels = [r"$ \hat{\nabla}_{\phi} h_k^T (Z_{1:K} - v_k)  $", r"$Z_{1:K} - v_k$", r"$Z_{1:K}$", r"$v_k$"]
            scores = [dqlogits_prod * (L_k - v), L_k - v, L_k, v]
            estimates = [dqlogits_prod * (L_hat - v_hat), L_hat - v_hat, L_hat, v_hat]

            for i, (key, label, score, estimate) in enumerate(zip(keys, labels, scores, estimates)):

                # print(f"{c_name}: {label}: score = {score.shape}, estimate = {estimate.shape}")
                score = score.contiguous().view(-1).data.cpu().numpy().tolist()
                estimate = estimate.contiguous().view(-1).data.cpu().numpy().tolist()
                assert len(score) == len(estimate), f"{c_name}: score = {len(score)}, estimate = {len(estimate)}"
                for index, (s, e) in enumerate(zip(score, estimate)):
                    data += [{'estimator': c_name, 'label': label, 'score': s, _key: np.abs(e - s), 'key':key, 'seed': index}]

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

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        for i, label in enumerate(labels):

            _data = data[data["label"] == label]

            # filter outliers
            values = _data[_key].values
            a, b = np.percentile(values, [25, 75])
            y_b = b + k_out * (b - a)
            y_a = a - k_out * (b - a)
            _data = _data[(_data[_key] <= y_b) & (_data[_key] >= y_a)]

            for c_name in c_names:
                c_data = _data[_data["estimator"] == c_name]

                values = c_data[_key].values

                ax = axes[1, i]
                sns.distplot(values, ax=ax, label=c_name, rug=False, kde=True, bins=100)

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

            if i < len(labels) - 1:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

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



def analyse_control_variate(x, model, c_configs, c_estimators, c_names, writer_train, seed=None, global_step=None, nsamples=1000):

    keys = ["a", "b", "c", "d"]
    labels = [r"$ \hat{\nabla}_{\phi} \sum_k h_k^T (Z_{1:K} - v_k)  $", r"$ \sum_k Z_{1:K} - v_k$",
              r"$ \sum_k Z_{1:K}$", r"$ \sum_k v_k$"]

    x = x[0][None].expand(nsamples, *x.size()[1:])

    _scores = dict()
    _estimates = dict()
    _dlogit_hats = dict()

    for c_onf, c_est, c_name in zip(c_configs, c_estimators, c_names):

        print("- analysis:", c_name)

        if seed is not None:
            _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(seed)

        bs, mc, iw = x.size(0), c_est.mc, c_est.iw
        loss, diagnostics, output = c_est(model, x, **c_onf)

        # comute the estimate of the grads using both estimators

        if "L_k" in output.keys() and "v" in output.keys():

            # get score
            L_k = output["L_k"][:, :, None].expand(bs, mc, iw)
            v = output["v"].expand(bs, mc, iw)
            dqlogits = output["dqlogits"].view(bs, mc, iw, -1)

            # get estimates
            L_hat = output.get("L_hat", torch.zeros_like(L_k)[..., None])
            v_hat = output.get("v_hat", torch.zeros_like(v)[..., None])

            # compute an estimate of the gradients for each sample
            dqlogits_hat = (dqlogits * (L_k[..., None] - v[..., None] - L_hat - v_hat))

            L_hat = L_hat.mean(-1)
            v_hat = v_hat.mean(-1)

            # print(f">> {c_name}: L_k = {L_k.shape}, L_hat = {output['L_hat'].shape} | {L_hat.shape}, v = {v.shape}, v_hat = {v_hat.shape}")

            scores = [L_k - v, L_k, v]
            estimates = [L_hat - v_hat, L_hat, v_hat]

            _scores[c_name] = scores
            _estimates[c_name] = estimates
            _dlogit_hats[c_name] = dqlogits_hat

    # conpute expected gradients based on all estimators
    #global_dqlogits_hat = torch.cat(list(_dlogit_hats.values()), 0).mean(dim=(0, 1), keepdim=True)


    data = []
    for c_name in c_names:
        global_dqlogits_hat = _dlogit_hats[c_name].mean(dim=(0, 1), keepdim=True)
        dqlogits_dot = (_dlogit_hats[c_name] * global_dqlogits_hat).sum(-1)

        print(c_name, global_dqlogits_hat.shape, dqlogits_dot.shape,  _scores[c_name][0].shape) # todo: DEBUG

        scores = [dqlogits_dot * _scores[c_name][0]] + _scores[c_name]
        estimates = [dqlogits_dot * _estimates[c_name][0]] + _estimates[c_name]

        for i, (key, label, score, estimate) in enumerate(zip(keys, labels, scores, estimates)):

            # print(f"{c_name}: {label}: score = {score.shape}, estimate = {estimate.shape}")
            score = score.contiguous().sum(2).view(-1).data.cpu().numpy().tolist()
            estimate = estimate.contiguous().sum(2).view(-1).data.cpu().numpy().tolist()
            assert len(score) == len(estimate), f"{c_name}: score = {len(score)}, estimate = {len(estimate)}"
            for index, (s, e) in enumerate(zip(score, estimate)):
                data += [
                    {'estimator': c_name, 'label': label, 'score': s, _key: np.abs(e - s), 'key': key, 'seed': i}]


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

            # ys rescaling
            ys = _data[_key].values.tolist()
            if label != r"$v_k$":
                a, b = np.percentile(ys, [25, 75])
                y_b = b + k_out * (b - a)
                y_a = a - k_out * (b - a)
                ax.set_ylim([y_a, y_b])

            # xs rescaling
            xs = _data["score"].values.tolist()
            if label != r"$v_k$":
                a, b = np.percentile(xs, [25, 75])
                x_b = b + k_out * (b - a)
                x_a = a - k_out * (b - a)
                ax.set_xlim([x_a, x_b])


            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        for i, label in enumerate(labels):

            _data = data[data["label"] == label]

            # filter outliers
            values = _data[_key].values

            a, b = np.percentile(values, [25, 75])
            y_b = b + k_out * (b - a)
            y_a = a - k_out * (b - a)
            filtered_data = _data[(_data[_key] <= y_b) & (_data[_key] >= y_a)]

            for k, c_name in enumerate(c_names):
                c_data = filtered_data[filtered_data["estimator"] == c_name]

                c_values = c_data[_key].values

                _mean = np.mean(_data[_data["estimator"] == c_name][_key].values)
                _std = np.std(_data[_data["estimator"] == c_name][_key].values)

                ax = axes[1, i]

                sns.distplot(c_values, ax=ax, label=c_name, rug=False, kde=False, bins=64)

                ax.axvline(x=_mean, color=colors[k], alpha = 0.7, linestyle="-")
                if _mean + 0.5 * _std <= y_b:
                    ax.axvline(x=_mean + 0.5 * _std, color=colors[k], alpha=0.7, linestyle=":")
                if _mean - 0.5 * _std >= 0:
                    ax.axvline(x=_mean - 0.5 * _std, color=colors[k], alpha=0.7, linestyle=":")


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

            if i < (len(labels) - 1):
                legend = axes[1, i].get_legend()
                if legend is not None:
                    legend.remove()

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