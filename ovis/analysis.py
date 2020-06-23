import sys
from typing import *

import torch


def preprocess(batch, device):
    if isinstance(batch, torch.Tensor):
        x = batch.to(device)
        return x, None
    else:
        x, y = batch  # assume tuple (x,y)
        x = x.to(device)
        y = y.to(device)
        return x, y


@torch.no_grad()
def latent_activations(model, loader, mc_samples, nsamples=None, epsilon=1e-3, seed=None):
    """
    Computing the activations of the latent variables
    """

    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    zs = None
    model.eval()
    device = next(iter(model.parameters())).device

    for batch in loader:
        x, *_ = preprocess(batch, device)
        # expand `x` for `mc_samples`
        bs, *dims = x.shape
        x = x.view(bs, 1, *dims).expand(bs, mc_samples, *dims).contiguous().view(-1, *dims)

        # forward pass
        output = model(x, tau=0)
        z = output['z']

        # concatenate as [bs x mc, *]
        def _reduce_z(x):
            return x.view(x.size(0), -1)

        if isinstance(z[0], Tuple):
            # quick 'n' dirty: handle SBM case
            z = z[0]

        if isinstance(z, List) or isinstance(z, Tuple):
            z = torch.cat([_reduce_z(zz) for zz in z], 1)

        # E_q(z|x) [z_d]
        z = z.view(bs, mc_samples, -1)

        z = z.mean(1)

        z = z.cpu()
        if zs is None:
            zs = z
        else:
            zs = torch.cat([zs, z], 0)

        if nsamples is not None:
            if zs.size(0) > nsamples:
                zs = zs[:nsamples]
                break

    # Cov_p(x) [ E_q(z|x) [z_d] ]
    N = zs.size(0)
    cov_expected_z_given_x = 1 / (N - 1) * torch.sum((zs - zs.mean(0, keepdim=True)).pow(2), dim=0)

    # AU = \sum_d \delta{ Cov_p(x) [ E_q(z|x) [z_d] ] > \epsilon }
    au = (cov_expected_z_given_x > epsilon).float().sum()

    if seed is not None:
        torch.manual_seed(_seed)

    return {'active_units': {'au': au,
                             'r_au': au / cov_expected_z_given_x.shape[0],
                             'n': cov_expected_z_given_x.shape[0]}}

