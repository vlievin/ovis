import sys
from time import time

import torch
from tqdm import tqdm

eps = 1e-15
min_var = 1e-9


def covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def cosine(u, v, dim=-1):
    return (u * v).sum(dim=dim) / (u.norm(dim=dim, p=2) * v.norm(dim=dim, p=2))


def _percentile(x, q=0.5):
    if x is not None:
        assert q < 1
        x = x.view(-1)
        k = int(q * x.shape[0]) + 1
        v, idx = x.kthvalue(k)  # indexing from 1..
        return v


def _mean(x):
    if x is not None:
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


def get_grads_from_tensor(model, loss, output, tensor_id, mc, iw):
    assert tensor_id in output.keys(), f"Tensor_id = `{tensor_id}` not in model's output"

    model.zero_grad()
    loss.sum().backward(create_graph=True, retain_graph=True)

    # get the tensor of interest
    tensors = output[tensor_id] if isinstance(output[tensor_id], list) else output[tensor_id]
    bs = tensors[0].shape[0] // (mc * iw)

    # get the gradients, flatten and concat across the feature dimension
    gradients = [p.grad for p in tensors]
    assert not any(
        [g is None for g in gradients]), f"{sum([int(g is None) for g in gradients])} tensors have no gradients." \
                                         f"Use `tensor.retain_graph()` in your model to enable gradients."

    # compute gradients estimate for each individual grads
    # sum individual gradients because x_expanded = x.expand(bs, mc, iw)
    gradients = torch.cat([g.view(bs, mc * iw, -1).sum(1) for g in gradients], 1)

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


def get_individual_gradients_statistics(estimator, model, x, batch_size=32, n_samples=100, seed=None,
                                        key_filter='qlogits',
                                        true_grads=None, return_grads=False, use_dsnr=False, **config):
    """
    Compute the variance, magnitude and SNR of the gradients for each data point. Each statistics is averaged on the values given by each point.
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
    grads_dsnr = Mean()
    grads_mean = Mean()
    grads_variance = Mean()
    if true_grads is not None:
        grads_dir = Mean()

    all_grads = None
    grads_expected = None

    with tqdm(total=x.shape[0] * iterations) as pbar:

        for i, x_i in enumerate(x):
            x_i = x_i[None].expand(effective_batch_size, *x.size()[1:]).contiguous()

            grads_mean_i = Mean()
            grads_variance_i = Variance()
            all_grads_i = None

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
                    gradients = get_grads_from_tensor(model, loss, output, tensor_id, estimator.mc, estimator.iw)
                else:
                    gradients = get_grads_from_parameters(model, loss, key_filter=key_filter)

                gradients = list(gradients)

                # update statistics
                for grads in gradients:
                    with torch.no_grad():
                        grads = grads.detach()

                        if return_grads or use_dsnr:
                            all_grads_i = grads[None] if all_grads_i is None else torch.cat([all_grads_i, grads[None]],
                                                                                            0)

                        grads_mean_i.update(grads)
                        grads_variance_i.update(grads)

                        if grads_mean_i.n >= n_samples:
                            break

            assert (grads_variance_i.n == n_samples) and (grads_mean_i.n == n_samples)

            with torch.no_grad():
                if return_grads:
                    all_grads = all_grads_i[None, :, :] if all_grads is None else torch.cat(
                        [all_grads, all_grads_i[None]],
                        0)

                # compute statistics for each data point `x_i`
                grads_variance_i = grads_variance_i()
                grads_mean_i = grads_mean_i()

                # expected value
                grads_expected = grads_mean_i[None] if grads_expected is None else torch.cat(
                    [grads_expected, grads_mean_i[None]], 0)

                # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
                grads_snr_i = grads_mean_i.abs() / (eps + grads_variance_i ** 0.5)

                # compute DSNR,  see `tighter variational bounds are not necessarily better` (eq. 12)
                if use_dsnr:
                    u = all_grads_i.mean(0, keepdim=True)
                    u /= u.norm(dim=1, keepdim=True, p=2)

                    g_parallel = u * (u * all_grads_i).sum(1, keepdim=True)
                    g_perpendicular = all_grads_i - g_parallel

                    dsnr_i = g_parallel.norm(dim=1, p=2) / (eps + g_perpendicular.norm(dim=1, p=2))

                    grads_dsnr.update(dsnr_i)

                # update global statistics
                grads_mean.update(grads_mean_i)
                grads_variance.update(grads_variance_i)
                grads_snr.update(grads_snr_i)
                if true_grads is not None:
                    cosine_grads_i = cosine(grads_mean_i, true_grads[i])
                    grads_dir.update(cosine_grads_i)

    print(f">>> elapsed time = {time() - _start:.3f}, estimator = {type(estimator).__name__}")

    # reinitialize grads
    model.zero_grad()

    # get global statistics
    grads_mean = grads_mean()
    grads_variance = grads_variance()
    grads_snr = grads_snr()
    grads_dsnr = grads_dsnr()
    if true_grads is not None:
        grads_dir = grads_dir()

    if seed is not None:
        torch.manual_seed(_seed)

    # reduce fn: keep only parameter with variance > 0
    mask = (grads_variance > min_var).float()
    _reduce = lambda x: (x * mask).sum() / mask.sum()

    output = {'grads': {
        'variance': _reduce(grads_variance),
        'magnitude': _reduce(grads_mean.abs()),
        'snr': _reduce(grads_snr),
        'dsnr': grads_dsnr.mean() if grads_dsnr is not None else 0.,
        'direction': grads_dir.mean() if true_grads is not None else 0.
    },
        'snr': {
            'p25': _percentile(grads_snr, q=0.25), 'p50': _percentile(grads_snr, q=0.50),
            'p75': _percentile(grads_snr, q=0.75), 'p5': _percentile(grads_snr, q=0.05),
            'p95': _percentile(grads_snr, q=0.95), 'min': grads_snr.min(),
            'max': grads_snr.max(), 'mean': grads_snr.mean()}
    }

    # additional data: raw grads, and mean,var,snr for each parameter separately
    meta = {
        'expected': grads_expected,
        'grads': all_grads,
        'magnitude': grads_mean.abs(),
        'var': grads_variance,
        'snr': grads_snr,
    }

    return output, meta


"""Version for gradients averaged over an entire batch"""


def get_batch_grads_from_tensor(model, loss, output, tensor_id, mc, iw):
    assert tensor_id in output.keys(), f"Tensor_id = `{tensor_id}` not in model's output"

    model.zero_grad()
    loss.sum().backward(create_graph=True, retain_graph=True)

    # get the tensor of interest
    tensors = output[tensor_id] if isinstance(output[tensor_id], list) else output[tensor_id]
    bs = tensors[0].shape[0] // (mc * iw)
    # get the gradients, flatten and concat across the feature dimension
    gradients = [p.grad for p in tensors]
    assert not any(
        [g is None for g in gradients]), f"{sum([int(g is None) for g in gradients])} tensors have no gradients. " \
                                         f"Use `tensor.retain_graph()` in your model to enable gradients. " \
                                         f"tensor_id = `{tensor_id}`"

    # compute gradients estimate for each individual grads
    # sum individual gradients because x_expanded = x.expand(bs, mc, iw)
    gradients = torch.cat([g.view(bs, mc * iw, -1).sum(1) for g in gradients], 1)

    # return each MC average of the grads
    return gradients.mean(0)


def get_batch_grads_from_parameters(model, loss, key_filter=''):
    params = [p for k, p in model.named_parameters() if key_filter in k]
    assert len(params) > 0, f"`No parameter matching the filter `{key_filter}`"
    model.zero_grad()
    # backward individual gradients \nabla L[i]
    loss.mean().backward(create_graph=True, retain_graph=True)
    # gather gradients for each parameter and concat such that each element across the dim 1 is a parameter
    grads = [p.grad.view(-1) for p in params if p.grad is not None]
    return torch.cat(grads, 0)


def get_batch_gradients_statistics(estimator, model, x, n_samples=100, seed=None, key_filter='qlogits',
                                   true_grads=None, return_grads=False, use_dsnr=False, **config):
    """
    Compute the variance, magnitude and SNR of the gradients averaged over a batch of data.
    """

    _start = time()

    # set specific seed
    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    # init statistics for each datapoint
    grads_dsnr = None
    grads_mean = Mean()
    grads_variance = Variance()
    if true_grads is not None:
        grads_dir = Mean()

    all_grads = None

    for i in tqdm(range(n_samples), desc="Batch Gradients Analysis"):

        model.eval()
        model.zero_grad()

        # forward, backward to compute the gradients
        loss, diagnostics, output = estimator(model, x, backward=False, **config)

        # gather individual gradients
        if 'tensor:' in key_filter:
            tensor_id = key_filter.replace("tensor:", "")
            gradients = get_batch_grads_from_tensor(model, loss, output, tensor_id, estimator.mc, estimator.iw)
        else:
            gradients = get_batch_grads_from_parameters(model, loss, key_filter=key_filter)

        # gather statistics
        with torch.no_grad():
            gradients = gradients.detach()

            if return_grads or use_dsnr:
                all_grads = gradients[None] if all_grads is None else torch.cat([all_grads, gradients[None]], 0)

            grads_mean.update(gradients)
            grads_variance.update(gradients)

    # compute statistics
    with torch.no_grad():

        # compute statistics for each data point `x_i`
        grads_variance = grads_variance()
        grads_mean = grads_mean()

        # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
        grads_snr = grads_mean.abs() / (eps + grads_variance ** 0.5)

        # compute DSNR,  see `tighter variational bounds are not necessarily better` (eq. 12)
        if use_dsnr:
            u = all_grads.mean(0, keepdim=True)
            u /= u.norm(dim=1, keepdim=True, p=2)

            g_parallel = u * (u * all_grads).sum(1, keepdim=True)
            g_perpendicular = all_grads - g_parallel

            grads_dsnr = g_parallel.norm(dim=1, p=2) / (eps + g_perpendicular.norm(dim=1, p=2))

        # compute grd direction
        if true_grads is not None:
            grads_dir = cosine(grads_mean, true_grads, dim=-1)

    print(
        f">>> elapsed time = {time() - _start:.3f}, estimator = {type(estimator).__name__}, K = {estimator.iw * estimator.mc}")

    # reinitialize grads
    model.zero_grad()

    if seed is not None:
        torch.manual_seed(_seed)

    # reduce fn: keep only parameter with variance > 0
    mask = (grads_variance > min_var).float()
    _reduce = lambda x: (x * mask).sum() / mask.sum()

    output = {'grads': {
        'variance': _reduce(grads_variance),
        'magnitude': _reduce(grads_mean.abs()),
        'snr': _reduce(grads_snr),
        'dsnr': grads_dsnr.mean() if grads_dsnr is not None else 0.,
        'direction': grads_dir.mean() if true_grads is not None else 0.
    },
        'snr': {
            'p25': _percentile(grads_snr, q=0.25), 'p50': _percentile(grads_snr, q=0.50),
            'p75': _percentile(grads_snr, q=0.75), 'p5': _percentile(grads_snr, q=0.05),
            'p95': _percentile(grads_snr, q=0.95), 'min': grads_snr.min(),
            'max': grads_snr.max(), 'mean': grads_snr.mean()}
    }

    # additional data: raw grads, and mean,var,snr for each parameter separately
    meta = {
        'grads': all_grads,
        'expected': grads_mean,
        'magnitude': grads_mean.abs(),
        'var': grads_variance,
        'snr': grads_snr,
    }

    return output, meta


def get_gradients_statistics(*args, use_individual_grads=False, **kwargs):
    if use_individual_grads:
        return get_individual_gradients_statistics(*args, **kwargs)
    else:
        return get_batch_gradients_statistics(*args, **kwargs)
