import sys

import numpy as np
import torch

eps = 1e-20


def covariance(x):
    x = x - x.mean(dim=1, keepdim=True)
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def get_gradients_statistics(estimator, model, x, batch_size=32, seed=None, **config):
    """
    Compute the variance, magnitude and SNR of the gradients.
    """

    all_grads = None
    control_variate_l1s = []

    for i, x_i in enumerate(x):
        x_i = x_i[None].expand(batch_size, *x_i.size())

        if seed is not None:
            _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(seed)

        model.eval()

        # forward, backward to compute the gradients
        loss, diagnostics, output = estimator(model, x_i, backward=False, **config)

        for j, l in enumerate(loss):

            model.zero_grad()
            l.mean().backward(create_graph=True, retain_graph=True)

            grads = torch.cat([p.grad.view(1, -1) for p in model.parameters() if p.grad is not None], 1)

            with torch.no_grad():
                if all_grads is None:
                    all_grads = grads
                else:
                    all_grads = torch.cat([all_grads, grads], 0)

        # return L1 term
        l1 = diagnostics.get('loss').get('control_variate_l1')
        control_variate_l1s += [l1.mean().item() if l1 is not None else 0.]

    if seed is not None:
        torch.manual_seed(_seed)

    # reinitialize grads
    model.zero_grad()

    avg_variance = all_grads.var(0).mean()
    avg_magnitude = all_grads.mean(0).abs().mean()
    avg_snr = (all_grads.mean(0) / (eps + all_grads.std(0))).abs().mean()
    avg_l1 = np.mean(control_variate_l1s)

    return {'log_variance': avg_variance.log(), 'magnitude': avg_magnitude, 'log_snr': avg_snr.log(),
            'reinforce_l1': avg_l1}


def get_gradients_log_total_variance__(estimator, model, x, batch_size=32, seed=None, **config):
    """
    Compute the average log of the total variance

    y = E_x[ trace( E_q(z|x) [ cov(grads(L(x, z)) ] ) ]
    """

    var_grads = []
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
        q_logits = [p for i, p in enumerate(output['qlogits'])]

        # get the gradients, flatten and concat them
        bs = x_i.size(0)
        gradients = torch.cat([p.grad.view(bs, -1) for p in q_logits], 1)

        with torch.no_grad():
            # compute the covariance of the gradients and the total variance
            gradients_covariance = covariance(gradients)
            total_variance = gradients_covariance.trace()

            # x_i output
            control_variate_l1 = diagnostics.get('loss').get('control_variate_l1')
            control_variate_l1s += [control_variate_l1.mean().item() if control_variate_l1 is not None else 0.]
            var_grads += [(total_variance).item()]

    if seed is not None:
        torch.manual_seed(_seed)

    # reinitialize grads
    model.zero_grad()

    return np.log(np.sum(var_grads) / len(var_grads)), np.mean(control_variate_l1s)
