import sys
from time import time

import numpy as np
import torch

eps = 1e-18


def covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def get_gradients_statistics__(estimator, model, x, batch_size=32, seed=None, key_filter='', **config):
    """
    Compute the variance, magnitude and SNR of the gradients.
    """

    _start = time()

    control_variate_l1s = []
    magnitudes = []
    variances = []

    for i, x_i in enumerate(x):
        # repeat sample x_i
        x_i = x_i[None].expand(batch_size, *x_i.size())

        if seed is not None:
            _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(seed)

        model.eval()

        # forward, backward to compute the gradients
        loss, diagnostics, output = estimator(model, x_i, backward=False, **config)

        # gather individual gradients
        all_grads = None
        for j, l in enumerate(loss):

            model.zero_grad()

            # backward individual gradients \nabla L[i]
            l.mean().backward(create_graph=True, retain_graph=True)

            # gather gradients for each parameter
            grads = torch.cat(
                [p.grad.view(1, -1) for k, p in model.named_parameters() if (p.grad is not None) and (key_filter in k)]
                , 1)

            # conctatenate individual grads into a batch of individual grads resulting in a tensor `all_grads` = `nsamples x params`
            with torch.no_grad():
                if all_grads is None:
                    all_grads = grads
                else:
                    all_grads = torch.cat([all_grads, grads], 0)

        with torch.no_grad():
            # return reinforce l1 term
            l1 = diagnostics.get('reinforce', {}).get('l1')
            control_variate_l1s += [l1.mean().item() if l1 is not None else 0.]

            # gather grads expected value and variance of the gradients
            variances += [all_grads.var(0, keepdim=True)]
            magnitudes += [all_grads.mean(0, keepdim=True)]

    if seed is not None:
        torch.manual_seed(_seed)

    # reinitialize grads
    model.zero_grad()

    # concatenate across x_i`s
    variances = torch.cat(variances)
    magnitudes = torch.cat(magnitudes)

    # compute batch variance and magnitude
    # see `tighter variational bounds are not necessarily better` (eq. 10)
    variance = (variances / variances.shape[0] ** 2).sum(0)
    magnitude = magnitudes.mean(0)

    # compute mask to exclude parameters with zero variance
    mask = (variance.pow(.5) > eps).float()

    # don't use mask for now
    mask = torch.ones_like(mask)

    def _mean(x):
        return (mask * x).sum() / mask.sum()

    # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
    snr = (magnitude / (eps + variance.pow(.5))).abs()
    # snr mean of NON zeros

    avg_l1 = np.mean(control_variate_l1s)

    # print(f">> grads: iw = {estimator.iw}, elapsed time = {time() - _start:.3f}, snr = {snr.mean().log().item():.3f}, masked. snr {_mean(snr).log().item():.3f},  log_var = {_mean(variance).log().item():.3f}, Estimator = {type(estimator).__name__}")

    return {'variance': _mean(variance), 'magnitude': _mean(magnitude).abs(), 'snr': _mean(snr),
            'reinforce_l1': avg_l1}


def get_gradients_statistics(estimator, model, x, batch_size=32, seed=None, key_filter='', **config):
    """
    Compute the average log of the total variance

    y = E_x[ trace( E_q(z|x) [ cov(grads(L(x, z)) ] ) ]
    """

    var_grads = []
    mean_grads = []
    control_variate_l1s = []

    for x_i in x:
        x_i = x_i[None].expand(x.size(0), *x_i.size())

        if seed is not None:
            _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(seed)

        model.eval()
        model.zero_grad()

        # forward, backward to compute the gradients
        loss, diagnostics, output = estimator(model, x_i, backward=False, **config)
        loss.mean().backward(create_graph=True, retain_graph=True)

        # get the logits of the variational distributions
        q_logits = [p for i, p in enumerate(output['qlogits'])]

        # get the gradients, flatten and concat them
        bs = x_i.size(0)
        gradients = torch.cat([p.grad.view(bs, -1) for p in q_logits], 1)

        with torch.no_grad():
            # compute the covariance of the gradients and the total variance
            gradients_covariance = covariance(gradients)
            total_variance = gradients_covariance.trace()

            # x_i output
            control_variate_l1 = diagnostics.get('reinforce', {}).get('l1', None)
            control_variate_l1s += [control_variate_l1.mean().item() if control_variate_l1 is not None else 0.]
            var_grads += [(total_variance).item()]
            mean_grads += [(gradients.norm(p=2, dim=1)).mean().item()]

    if seed is not None:
        torch.manual_seed(_seed)

    # reinitialize grads
    model.zero_grad()

    variance = np.sum(var_grads) / len(var_grads) ** 2
    magnitude = np.mean(mean_grads)
    snr = magnitude / (eps + variance ** (0.5))

    avg_l1 = np.mean(control_variate_l1s)

    return {'variance': variance, 'magnitude': magnitude, 'snr': snr,
            'reinforce_l1': avg_l1}


def get_parameter_wise_grads(model, loss, key_filter=''):
    # gather individual gradients
    all_grads = None
    for j, l in enumerate(loss):

        model.zero_grad()

        # backward individual gradients \nabla L[i]
        l.mean().backward(create_graph=True, retain_graph=True)

        # gather gradients for each parameter
        grads = [p.grad.view(1, -1) for k, p in model.named_parameters() if (p.grad is not None) and (key_filter in k)]
        grads = torch.cat(grads, 1)

        # conctatenate individual grads into a batch of individual grads resulting in a tensor `all_grads` = `nsamples x params`
        with torch.no_grad():
            if all_grads is None:
                all_grads = grads
            else:
                all_grads = torch.cat([all_grads, grads], 0)

    return all_grads


def get_l1(diagnostics):
    control_variate_l1 = diagnostics.get('reinforce', {}).get('l1', None)
    return control_variate_l1.mean().item() if control_variate_l1 is not None else 0.


class Mean():
    def __init__(self):
        self.mean = None
        self.n = 0

    def update(self, x):
        self.n += 1
        if self.mean is None:
            self.mean = x
        else:
            self.mean = (self.n - 1) / self.n * self.mean + 1. / self.n * x

    def __call__(self):
        return self.mean


class Variance():
    def __init__(self):
        self.n = 0
        self.Ex = None
        self.Ex2 = None
        self.K = None

    def update(self, x):
        self.n += 1
        if self.K is None:
            self.K = x
            self.Ex = x - self.K
            self.Ex2 = (x - self.K) ** 2
        else:
            self.Ex += x - self.K
            self.Ex2 += (x - self.K) ** 2

    def __call__(self):
        return (self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1)


def get_gradients_statistics(estimator, model, x, batch_size=32, n_samples=1, seed=None, key_filter='', **config):
    """
    Compute the variance, magnitude and SNR of the gradients.
    """
    # if estimator.iw > 100:
    #     n_samples = n_samples * batch_size
    #     batch_size = 1

    _start = time()

    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    grads_snr = Mean()
    grads_mean = Mean()
    grads_variance = Mean()

    for x_i in x:
        x_i = x_i[None].expand(n_samples, *x.size()[1:])

        grads_mean_i = Mean()
        grads_variance_i = Variance()

        for i, x_j in enumerate(x_i):

            # repeat sample x_i
            x_j = x_j[None].expand(batch_size, *x.size()[1:]).contiguous()

            model.eval()
            model.zero_grad()

            # forward, backward to compute the gradients
            loss, diagnostics, output = estimator(model, x_j, backward=False, **config)

            if 'qlogits' == key_filter:
                loss.mean().backward(create_graph=True, retain_graph=True)

                # get the logits of the variational distributions
                q_logits = [p for i, p in enumerate(output['qlogits'])]

                # get the gradients, flatten and concat them
                bs = x_j.size(0)
                gradients = torch.cat([p.grad.view(bs, -1) for p in q_logits], 1)

                for grads in gradients:
                    # conctatenate individual grads into a batch of individual grads resulting in a tensor `all_grads` = `nsamples x params`
                    with torch.no_grad():
                        grads = grads.detach()
                        grads_mean_i.update(grads)
                        grads_variance_i.update(grads)

            else:
                for j, l in enumerate(loss):

                    model.zero_grad()

                    # backward individual gradients \nabla L[i]
                    l.mean().backward(create_graph=True, retain_graph=True)

                    # gather gradients for each parameter
                    grads = [p.grad.view(-1) for k, p in model.named_parameters() if
                             (p.grad is not None) and (key_filter in k)]
                    grads = torch.cat(grads, 0)

                    # conctatenate individual grads into a batch of individual grads resulting in a tensor `all_grads` = `nsamples x params`
                    with torch.no_grad():
                        grads = grads.detach()
                        grads_mean_i.update(grads)
                        grads_variance_i.update(grads)

        grads_variance_i = grads_variance_i()
        grads_mean_i = grads_mean_i()

        # filter
        mask = grads_variance_i > 0
        grads_variance_i = grads_variance_i[mask]
        grads_mean_i = grads_mean_i[mask]

        # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
        grads_snr_i = grads_mean_i.abs() / grads_variance_i**(0.5)

        # update average
        grads_mean.update(grads_variance_i)
        grads_variance.update(grads_variance_i)
        grads_snr.update(grads_snr_i)

    # reinitialize grads
    model.zero_grad()


    # get statistics
    grads_mean = grads_mean()
    grads_variance = grads_variance()
    grads_snr = grads_snr()

    def _reduce(x):
        # x = x.view(-1)
        # k = int(0.5 * x.shape[0])
        # v, idx = x.kthvalue(k)
        # return v
        return x.mean()

    # print(f">> grads: iw = {estimator.iw}, elapsed time = {time() - _start:.3f}, snr = {snr.mean().log().item():.3f}, masked. snr {_mean(snr).log().item():.3f},  log_var = {_mean(variance).log().item():.3f}, Estimator = {type(estimator).__name__}")

    if seed is not None:
        torch.manual_seed(_seed)

    return {'variance': _reduce(grads_variance), 'magnitude': _reduce(grads_mean.abs()), 'snr': _reduce(grads_snr), 'keep_ratio': mask.float().sum()/mask.shape[0]}
