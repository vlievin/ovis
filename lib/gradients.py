import math
import torch
from collections import defaultdict


def get_gradients_log_total_variance(estimator, model, x, **config):
    sum_var = 0
    model.train()
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    for l in loss:
        l.backward(create_graph=True, retain_graph=True)
        for k, v in model.named_parameters():
            if v.grad is not None:
                sum_var += v.grad.detach().var(0).sum().detach().item()

    return math.log(1e-18 + sum_var)