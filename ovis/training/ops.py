from time import time

import torch


def append_ellapsed_time(func):
    """append the elapsed time to the diagnostics"""
    def wrapper(*args, **kwargs):
        start_time = time()
        diagnostics = func(*args, **kwargs)
        diagnostics['info']['elapsed-time'] = time() - start_time
        return diagnostics

    return wrapper


@append_ellapsed_time
def training_step(x, model, estimator, optimizers, **config):
    loss, diagnostics, output = estimator(model, x, backward=True, **config)

    [o.step() for o in optimizers]
    [o.zero_grad() for o in optimizers]

    return diagnostics


@torch.no_grad()
@append_ellapsed_time
def test_step(x, model, estimator, **config):
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    return diagnostics
