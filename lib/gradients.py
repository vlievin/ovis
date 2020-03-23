import sys
import numpy as np
import torch
from collections import defaultdict
from .utils import flatten, print_summary

eps = 1e-20

def covariance(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def get_gradients_log_total_variance(estimator, model, x, batch_size=32, seed=None, **config):
    """
    Compute the average log of the total variance

    y = E_x[ trace( E_q(z|x) [ cov(grads(L(x, z)) ] ) ]
    """

    log_sum_var_grads = []
    control_variate_l1s = []

    for x_i in x:
        x_i = x_i[None].expand(batch_size, *x_i.size())

        if seed is not None:
            _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(seed)

        model.eval()
        model.zero_grad()

        # forward, backward to compute the gradients
        loss, diagnostics, output = estimator(model, x_i, backward=False, **config)
        loss.mean().backward(create_graph=True, retain_graph=True)

        # get the logits of the variational distributions
        q_logits = [ p for i, p in enumerate(output['qlogits'])]

        # get the gradients, flatten and concat them
        bs = x_i.size(0)
        gradients = torch.cat([p.grad.view(bs, -1) for p in q_logits], 1)

        with torch.no_grad():
            # compute the covariance of the gradients and the total variance
            gradients_covariance = covariance(gradients)
            total_variance = gradients_covariance.trace()

            # x_i output
            control_variate_l1 = diagnostics.get('loss').get('control_variate_l1')
            control_variate_l1s += [ control_variate_l1.mean().item() if control_variate_l1 is not None else 0. ]
            log_sum_var_grads += [(eps + total_variance).log().item()]

    if seed is not None:
        torch.manual_seed(_seed)

    # reinitialize grads
    model.zero_grad()

    # print_summary(torch.tensor(log_sum_var_grads), "log_sum_var_grads")

    return np.mean(log_sum_var_grads), np.mean(control_variate_l1s)
