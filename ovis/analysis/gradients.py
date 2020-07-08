from time import time
from typing import *

import torch
from booster import Diagnostic
from torch import Tensor
from tqdm import tqdm

from .utils import cosine, percentile, RunningMean, RunningVariance
from ..estimators import GradientEstimator
from ..models import TemplateModel


def get_grads_from_tensor(model: TemplateModel, loss: Tensor, output: Dict[str, Tensor], tensor_id: str, mc: int, iw: int):
    """
    Compute the gradients given a `tensor` on which was called `tensor.retain_graph()`
    Assumes `tensor` to have `tensor.shape[0] == bs * iw * mc`

    :param model: VAE model
    :param loss: loss value
    :param output: model's output: dict
    :param tensor_id: key of the tensor in the model output
    :param mc: number of outer Monte-Carlo samples
    :param iw: number of inner Importance-Weighted samples
    :return: gradient: Tensor of shape [D,] where D is the number of elements in `tensor`
    """
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

    # return an MC average of the grads
    return gradients.mean(0)


def get_grads_from_parameters(model: TemplateModel, loss: Tensor, key_filter: str = ''):
    """
    Return the gradients for the parameters matching the `key_filter`

    :param model: VAE model
    :param loss: loss value
    :param key_filter: filter value (comma separated values accepted (e.g. "A,b"))
    :return:  Tensor of shape [D,] where `D` is the number of parameters
    """
    key_filters = key_filter.split(',')
    params = [p for k, p in model.named_parameters() if any([(_key in k) for _key in key_filters])]
    assert len(params) > 0, f"No parameters matching filter = `{key_filters}`"
    model.zero_grad()
    # backward individual gradients \nabla L[i]
    loss.mean().backward(create_graph=True, retain_graph=True)
    # gather gradients for each parameter and concat such that each element across the dim 1 is a parameter
    grads = [p.grad.view(-1) for p in params if p.grad is not None]
    return torch.cat(grads, 0)


def get_gradients_statistics(estimator: GradientEstimator,
                             model: TemplateModel,
                             x: Tensor,
                             mc_samples: int = 100,
                             key_filter: str = 'inference_network',
                             oracle_grad: Optional[Tensor] = None,
                             return_grads: bool = False,
                             compute_dsnr: bool = True,
                             samples_per_batch: Optional[int] = None,
                             eps: float = 1e-15,
                             tqdm: Callable = tqdm,
                             **config: Dict) -> Tuple[Diagnostic, Dict]:
    """
    Compute the gradients and return the statistics (Variance, Magnitude, SNR, DSNR)
    If an `oracle` gradient is available: compute the cosine similarity with the oracle and the gradient estimate (direction)

    The Magnitude, Variance and SNR are defined parameter-wise. All return values are average over the D parameters with
    Variance > eps. For instance, the returned SNR is

      * SNR = 1/D \sum_d SNR_d

    Each MC sample is computed sequentially and the mini-batch `x` will be split into chuncks
    if a value `samples_per_batch` if specified and if `samples_per_batch < x.size(0) * mc * iw`.

    :param estimator: Gradient Estimator
    :param model: VAE model
    :param x: mini-batch of observations
    :param mc_samples: number of Monte-Carlo samples
    :param key_filter: key matching parameters names in the model 
    :param oracle_grad: true direction of the gradients [Optional]
    :param return_grads: return all gradients in the `meta` output directory if set to `True`
    :param compute_dsnr: compute the Directional SNR if set to `True`
    :param samples_per_batch: max. number of individual samples `bs * mc * iw` per mini-batch [Optional]
    :param eps: minimum Variance value used for filtering
    :param config: config dictionary for the estimator
    :param tqdm: custom `tqdm` function
    :return: output : Diagnostic = {'grads' : {'variance': ..,
                                               'magnitude': ..,
                                               'snr': ..,
                                               'dsnr' ..,
                                               'direction': cosine similarity with the oracle,
                                               'keep_ratio' : ratio of parameter-wise gradients > epsilon}}
                                    'snr': {'percentiles', 'mean', 'min', 'max'}

                                    },
            meta : additional data including the gradients values if `return_grads`
    """
    _start = time()
    grads_dsnr = None
    grads_mean = RunningMean()
    grads_variance = RunningVariance()
    if oracle_grad is not None:
        grads_dir = RunningMean()

    all_grads = None

    # compute each MC sample sequentially
    for i in tqdm(range(mc_samples), desc="Gradients Analysis"):

        # compute number of chuncks based on the capacity `samples_per_batch`
        if samples_per_batch is None:
            chuncks = 1
        else:
            bs = x.size(0)
            mc = estimator.config['mc']
            iw = estimator.config['iw']
            # infer number of chunks
            total_samples = bs * mc * iw
            chuncks = max(1, -(-total_samples // samples_per_batch))  # ceiling division

        # compute mini-batch gradient by chunck if `x` is large
        gradients = RunningMean()
        for k, x_ in enumerate(x.chunk(chuncks, dim=0)):

            model.eval()
            model.zero_grad()

            # forward, backward to compute the gradients
            loss, diagnostics, output = estimator(model, x_, backward=False, **config)

            # gather mini-batch gradients
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

            if return_grads or compute_dsnr:
                all_grads = gradients[None] if all_grads is None else torch.cat([all_grads, gradients[None]], 0)

            grads_mean.update(gradients)
            grads_variance.update(gradients)

    # compute the statistics
    with torch.no_grad():

        # compute statistics for each data point `x_i`
        grads_variance = grads_variance()
        grads_mean = grads_mean()

        # compute signal-to-noise ratio. see `tighter variational bounds are not necessarily better` (eq. 4)
        grad_var_sqrt = grads_variance.pow(0.5)
        clipped_variance_sqrt = grad_var_sqrt.clamp(min=eps)
        grads_snr = grads_mean.abs() / (clipped_variance_sqrt)

        # compute DSNR,  see `tighter variational bounds are not necessarily better` (eq. 12)
        if compute_dsnr:
            u = all_grads.mean(0, keepdim=True)
            u /= u.norm(dim=1, keepdim=True, p=2)

            g_parallel = u * (u * all_grads).sum(1, keepdim=True)
            g_perpendicular = all_grads - g_parallel

            grads_dsnr = g_parallel.norm(dim=1, p=2) / (eps + g_perpendicular.norm(dim=1, p=2))

        # compute grad direction: cosine similarity between the gradient estimate and the oracle
        if oracle_grad is not None:
            grads_dir = cosine(grads_mean, oracle_grad, dim=-1)

    # reinitialize grads
    model.zero_grad()

    # reduce fn: keep only parameter with variance > 0
    mask = (grads_variance > eps).float()
    _reduce = lambda x: (x * mask).sum() / mask.sum()

    output = Diagnostic({'grads': {
        'variance': _reduce(grads_variance),
        'magnitude': _reduce(grads_mean.abs()),
        'snr': _reduce(grads_snr),
        'dsnr': grads_dsnr.mean() if grads_dsnr is not None else 0.,
        'keep_ratio': mask.sum() / torch.ones_like(mask).sum()
    },
        'snr': {
            'p25': percentile(grads_snr, q=0.25), 'p50': percentile(grads_snr, q=0.50),
            'p75': percentile(grads_snr, q=0.75), 'p5': percentile(grads_snr, q=0.05),
            'p95': percentile(grads_snr, q=0.95), 'min': grads_snr.min(),
            'max': grads_snr.max(), 'mean': grads_snr.mean()}
    })

    if oracle_grad is not None:
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
