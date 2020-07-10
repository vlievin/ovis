from functools import wraps
from time import time

import torch


def append_elapsed_time(func):
    """append the elapsed time to the diagnostics"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        diagnostics = func(*args, **kwargs)
        diagnostics['info']['elapsed-time'] = time() - start_time
        return diagnostics

    return wrapper


@append_elapsed_time
def training_step(x, model, estimator, optimizers, **config):
    """Perform one optimization step of the `model` given the mini-batch of observations `x`,
    the gradient estimator/evaluator `estimator` and the list of optimizers `optimizers`"""
    loss, diagnostics, output = estimator(model, x, backward=True, **config)

    [o.step() for o in optimizers]
    [o.zero_grad() for o in optimizers]

    return diagnostics


@torch.no_grad()
@append_elapsed_time
def test_step(x, model, estimator, **config):
    """Test the `model` given the mini-batch of observations `x` and the gradient estimator/evaluator `estimator`"""
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    return diagnostics
