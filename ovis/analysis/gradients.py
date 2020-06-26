from time import time

import torch
from tqdm import tqdm

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


class RunningMean():
    def __init__(self):
        self.mean = None
        self.n = 0

    def update(self, x, k=1):
        """use k > 1 if x is averaged over `k` points, k > 1"""

        if self.mean is None:
            self.mean = x
        else:
            self.mean = self.n / (self.n + k) * self.mean + k / (self.n + k) * x

        self.n += k

    def __call__(self):
        return self.mean


class RunningVariance():
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
        [g is None for g in gradients]), f"{sum([int(g is None) for g in gradients])} tensors have no gradients. " \
                                         f"Use `tensor.retain_graph()` in your model to enable gradients. " \
                                         f"tensor_id = `{tensor_id}`"

    # compute gradients estimate for each individual grads
    # sum individual gradients because x_expanded = x.expand(bs, mc, iw)
    gradients = torch.cat([g.view(bs, mc * iw, -1).sum(1) for g in gradients], 1)

    # return each MC average of the grads
    return gradients.mean(0)


def get_grads_from_parameters(model, loss, key_filter=''):
    key_filters = key_filter.split(',')
    params = [p for k, p in model.named_parameters() if any([(_key in k) for _key in key_filters])]
    assert len(params) > 0, f"No parameters matching filter = `{key_filters}`"
    model.zero_grad()
    # backward individual gradients \nabla L[i]
    loss.mean().backward(create_graph=True, retain_graph=True)
    # gather gradients for each parameter and concat such that each element across the dim 1 is a parameter
    grads = [p.grad.view(-1) for p in params if p.grad is not None]
    return torch.cat(grads, 0)


def get_gradients_statistics(estimator, model, x, n_samples=100, key_filter='qlogits',
                             true_grads=None, return_grads=False, use_dsnr=False, samples_per_batch=None,
                             eps=1e-18,
                             **config):
    """
    Compute the variance, magnitude and SNR of the gradients averaged over a batch of data.
    """

    _start = time()

    # init statistics for each datapoint
    grads_dsnr = None
    grads_mean = RunningMean()
    grads_variance = RunningVariance()
    if true_grads is not None:
        grads_dir = RunningMean()

    all_grads = None

    for i in tqdm(range(n_samples), desc="Batch Gradients Analysis"):

        # compute number of chuncks
        if samples_per_batch is None:
            chuncks = 1
        else:
            bs = x.size(0)
            mc = estimator.mc
            iw = estimator.iw
            # infer number of chunks
            total_samples = bs * mc * iw
            chuncks = max(1, -(-total_samples // samples_per_batch))  # ceiling division

        gradients = RunningMean()
        for k, x_ in enumerate(x.chunk(chuncks, dim=0)):

            model.eval()
            model.zero_grad()

            # forward, backward to compute the gradients
            loss, diagnostics, output = estimator(model, x_, backward=False, **config)

            # gather individual gradients
            if 'tensor:' in key_filter:
                tensor_id = key_filter.replace("tensor:", "")
                gradients_ = get_grads_from_tensor(model, loss, output, tensor_id, estimator.mc, estimator.iw)
            else:
                gradients_ = get_grads_from_parameters(model, loss, key_filter=key_filter)

            # move to cpu
            gradients_ = gradients_.detach().cpu()

            # update average
            gradients.update(gradients_, k=x_.size(0))

        # gather statistics
        with torch.no_grad():
            gradients = gradients()

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

    # reinitialize grads
    model.zero_grad()

    # reduce fn: keep only parameter with variance > 0
    mask = (grads_variance > eps).float()
    _reduce = lambda x: (x * mask).sum() / mask.sum()

    output = {'grads': {
        'variance': _reduce(grads_variance),
        'magnitude': _reduce(grads_mean.abs()),
        'snr': _reduce(grads_snr),
        'dsnr': grads_dsnr.mean() if grads_dsnr is not None else 0.,
        'keep_ratio': mask.sum() / torch.ones_like(mask).sum()
    },
        'snr': {
            'p25': _percentile(grads_snr, q=0.25), 'p50': _percentile(grads_snr, q=0.50),
            'p75': _percentile(grads_snr, q=0.75), 'p5': _percentile(grads_snr, q=0.05),
            'p95': _percentile(grads_snr, q=0.95), 'min': grads_snr.min(),
            'max': grads_snr.max(), 'mean': grads_snr.mean()}
    }

    if true_grads is not None:
        output['grads']['direction'] = grads_dir.mean()

    # additional data: raw grads, and mean,var,snr for each parameter separately
    meta = {
        'grads': all_grads,
        'expected': grads_mean,
        'magnitude': grads_mean.abs(),
        'var': grads_variance,
        'snr': grads_snr,
    }

    return output, meta
