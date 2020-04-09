import sys
from time import time

import torch
from tqdm import tqdm

eps = 1e-15


def covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def _percentile(x, q=0.5):
    assert q < 1
    x = x.view(-1)
    k = int(q * x.shape[0]) + 1
    v, idx = x.kthvalue(k)  # indexing from 1..
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


def get_grads_from_tensor(model, loss, output, tensor_id):
    assert tensor_id in output.keys(), f"Tensor_id = `{tensor_id}` not in model's output"

    model.zero_grad()
    loss.sum().backward(create_graph=True, retain_graph=True)

    # get the tensor of interest
    tensors = output[tensor_id] if isinstance(output[tensor_id], list) else output[tensor_id]
    bs = tensors[0].shape[0]
    # get the gradients, flatten and concat across the feature dimension
    gradients = [p.grad for p in tensors]
    assert not any(
        [g is None for g in gradients]), f"{sum([int(g is None) for g in gradients])} tensors have no gradients." \
                                         f"Use `tensor.retain_graph()` in your model to enable gradients."
    gradients = torch.cat([g.view(bs, -1) for g in gradients], 1)

    # return each individual gradient
    for grads in gradients:
        yield grads


def get_grads_from_parameters(model, loss, key_filter=''):
    params = [p for k, p in model.named_parameters() if key_filter in k]
    assert len(params) > 0, f"`No parameter matching the filter `{key_filter}`"
    for j, l in enumerate(loss):
        model.zero_grad()
        # backward individual gradients \nabla L[i]
        l.mean().backward(create_graph=True, retain_graph=True)
        # gather gradients for each parameter and concat such that each element across the dim 1 is a parameter
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        # return each individual gradient
        grads = torch.cat(grads, 0)
        yield grads


def get_gradients_statistics(estimator, model, x, batch_size=32, n_samples=100, seed=None, key_filter='qlogits', true_grads=None, **config):
    """
    Compute the variance, magnitude and SNR of the gradients.
    """

    n_individual_samples = n_samples * estimator.mc * estimator.iw
    effective_batch_size = max(1, batch_size // (estimator.mc * estimator.iw))
    iterations = n_samples // effective_batch_size + int(n_samples % effective_batch_size > 0)

    _start = time()

    # set specific seed
    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    # init statistics for each datapoint
    grads_snr = Mean()
    grads_mean = Mean()
    grads_variance = Mean()
    if true_grads is not None:
        grads_dir = Mean()
        true_grads = true_grads / true_grads.norm(p=2)

    with tqdm(total=x.shape[0] * iterations) as pbar:

        for i, x_i in enumerate(x):
            x_i = x_i[None].expand(effective_batch_size, *x.size()[1:]).contiguous()

            grads_mean_i = Mean()
            grads_variance_i = Variance()
            grads_dir_i = Mean() if true_grads is not None else None

            while grads_mean_i.n < n_samples:

                remaining = n_samples - grads_mean_i.n
                batch_size_i = min(remaining, effective_batch_size)

                pbar.set_description(f"gradients analysis [x = {i} / {x.shape[0]}, n = {grads_mean_i.n} / {n_samples}]")
                pbar.update(1)

                model.eval()
                model.zero_grad()

                # forward, backward to compute the gradients
                loss, diagnostics, output = estimator(model, x_i[:batch_size_i], backward=False, **config)

                # gather individual gradients
                if 'tensor:' in key_filter:
                    tensor_id = key_filter.replace("tensor:", "")
                    gradients = get_grads_from_tensor(model, loss, output, tensor_id)
                else:
                    gradients = get_grads_from_parameters(model, loss, key_filter=key_filter)

                gradients = list(gradients)

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

            # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
            grads_snr_i = grads_mean_i.abs() / (eps + grads_variance_i ** 0.5)

            # update global statistics
            grads_mean.update(grads_mean_i)
            grads_variance.update(grads_variance_i)
            grads_snr.update(grads_snr_i)
            if true_grads is not None:
                cosine_grads_i = (grads_mean_i * true_grads).sum() / grads_mean_i.norm(p=2)
                grads_dir.update(cosine_grads_i)

    print(f">>> elapsed time = {time() - _start:.3f}")

    # reinitialize grads
    model.zero_grad()

    # get global statistics
    grads_mean = grads_mean()
    grads_variance = grads_variance()
    grads_snr = grads_snr()
    if true_grads is not None:
        grads_dir = grads_dir()

    # print(f">> grads: iw = {estimator.iw}, elapsed time = {time() - _start:.3f}, snr = {snr.mean().log().item():.3f}, masked. snr {_mean(snr).log().item():.3f},  log_var = {_mean(variance).log().item():.3f}, Estimator = {type(estimator).__name__}")

    if seed is not None:
        torch.manual_seed(_seed)

    # reduce fn
    _reduce = _mean

    return {'grads': {
        'variance': _reduce(grads_variance),
        'magnitude': _reduce(grads_mean.abs()),
        'snr': _reduce(grads_snr),
        'direction': _reduce(grads_dir) if true_grads is not None else 0.
    },
        'snr': {
            'p25': _percentile(grads_snr, q=0.25), 'p50': _percentile(grads_snr, q=0.50),
            'p75': _percentile(grads_snr, q=0.75), 'p5': _percentile(grads_snr, q=0.05),
            'p95': _percentile(grads_snr, q=0.95), 'min': grads_snr.min(),
            'max': grads_snr.max(), 'mean': grads_snr.mean()}
    }
