import sys
from time import time

import torch
from tqdm import tqdm

eps = 1e-18


def covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def _median(x):
    x = x.view(-1)
    k = int(0.5 * x.shape[0])
    v, idx = x.kthvalue(k)
    return v


def _mean(x):
    return x.mean()


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


def get_grads_from_qlogits(model, loss, output):
    model.zero_grad()
    loss.mean().backward(create_graph=True, retain_graph=True)

    # get the logits of the variational distributions
    q_logits = [p for i, p in enumerate(output['qlogits'])]
    bs = q_logits[0].shape[0]
    # get the gradients, flatten and concat across the feature dimension
    gradients = torch.cat([p.grad.view(bs, -1) for p in q_logits], 1)
    # return each individual gradient
    for grads in gradients:
        yield grads


def get_grads_from_parameters(model, loss, key_filter=''):
    for j, l in enumerate(loss):
        model.zero_grad()
        # backward individual gradients \nabla L[i]
        l.mean().backward(create_graph=True, retain_graph=True)
        # gather gradients for each parameter and concat such that each element across the dim 1 is a parameter
        grads = [p.grad.view(-1) for k, p in model.named_parameters() if
                 (p.grad is not None) and (key_filter in k)]
        # return each individual gradient
        yield torch.cat(grads, 0)


def get_gradients_statistics(estimator, model, x, batch_size=32, n_samples=1000, seed=None, key_filter='', **config):
    """
    Compute the variance, magnitude and SNR of the gradients.
    """

    n_individual_samples = n_samples * estimator.mc * estimator.iw
    effective_batch_size = max(1, batch_size // (estimator.mc * estimator.iw))
    iterations = n_samples // effective_batch_size + int(n_samples % effective_batch_size > 0)

    print(f">>> Gradients: n_samples = {n_samples}, batch_size = {batch_size}, iw = {estimator.iw}")
    print(
        f">>> Gradients: n_individual_samples = {n_individual_samples}, effective_batch_size = {effective_batch_size}, iterations = {iterations}")

    _start = time()

    # set specific seed
    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    # init statistics for each datapoint
    grads_snr = Mean()
    grads_mean = Mean()
    grads_variance = Mean()

    with tqdm(total=x.shape[0] * iterations) as pbar:

        for i, x_i in enumerate(x):
            x_i = x_i[None].expand(effective_batch_size, *x.size()[1:]).contiguous()

            grads_mean_i = Mean()
            grads_variance_i = Variance()

            while grads_mean_i.n < n_samples:

                pbar.set_description(f"grad: x = {i} / {x.shape[0]}, n = {grads_mean_i.n} / {n_samples}")
                pbar.update(1)

                model.eval()
                model.zero_grad()

                # forward, backward to compute the gradients
                loss, diagnostics, output = estimator(model, x_i, backward=False, **config)

                # gather individual gradients
                if 'qlogits' == key_filter:
                    gradients = get_grads_from_qlogits(model, loss, output)
                else:
                    gradients = get_grads_from_parameters(model, loss, key_filter=key_filter)

                # update statistics
                for grads in gradients:
                    with torch.no_grad():
                        grads = grads.detach()
                        grads_mean_i.update(grads)
                        grads_variance_i.update(grads)

                        if grads_mean_i.n >= n_samples:
                            break

            assert (grads_variance_i.n == n_samples) and (grads_mean_i.n == n_samples)

            # compute statistics for each data point `x_i`
            grads_variance_i = grads_variance_i()
            grads_mean_i = grads_mean_i()

            # filter
            mask = grads_variance_i > 0
            grads_variance_i = grads_variance_i[mask]
            grads_mean_i = grads_mean_i[mask]

            # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
            grads_snr_i = grads_mean_i.abs() / grads_variance_i ** (0.5)

            # update global statistics
            grads_mean.update(grads_variance_i)
            grads_variance.update(grads_variance_i)
            grads_snr.update(grads_snr_i)

    print(f">>> elapsed time = {time() - _start:.3f}")

    # reinitialize grads
    model.zero_grad()

    # get global statistics
    grads_mean = grads_mean()
    grads_variance = grads_variance()
    grads_snr = grads_snr()

    # print(f">> grads: iw = {estimator.iw}, elapsed time = {time() - _start:.3f}, snr = {snr.mean().log().item():.3f}, masked. snr {_mean(snr).log().item():.3f},  log_var = {_mean(variance).log().item():.3f}, Estimator = {type(estimator).__name__}")

    if seed is not None:
        torch.manual_seed(_seed)

    _reduce = _mean
    return {'variance': _reduce(grads_variance), 'magnitude': _reduce(grads_mean.abs()), 'snr': _reduce(grads_snr),
            'keep_ratio': mask.float().sum() / mask.shape[0]}
