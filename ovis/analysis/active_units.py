from typing import *

import torch
from booster import Diagnostic
from torch.utils.data import DataLoader

from ..models import Template
from ..training.utils import preprocess


@torch.no_grad()
def latent_activations(model: Template,
                       loader: DataLoader,
                       mc_samples: int,
                       max_samples: Optional[int] = None,
                       epsilon: float = 1e-3) -> Diagnostic:
    """
    Computing the activations of the latent variables as in [https://arxiv.org/abs/1509.00519, https://arxiv.org/abs/1807.04863]:

      * au = \sum_{d=1}^D 1\{ Cov_{p(x)} [ E_{q(z|x)} [z_d] ] \}

    :param model: VAE model
    :param loader: data loader
    :param mc_samples: number of Monte Carlo samples
    :param max_samples: maximum number of data points
    :param epsilon: covariance threshold value
    :return: Diagnostic {'active_units' : {'au' number of active units: int,
                                           'r_au': ratio of active units: float}
                                           'n': total number of units: int} }
    """
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
            # quick 'n' dirty: handle the TVO models
            z = z[0]

        if isinstance(z, List) or isinstance(z, Tuple):
            z = torch.cat([_reduce_z(zz) for zz in z], 1)

        # E_q(z|x) [z_d]
        z = z.view(bs, mc_samples, -1).mean(1)

        # concatenate samples across `x`
        z = z.cpu()
        if zs is None:
            zs = z
        else:
            zs = torch.cat([zs, z], 0)

        if max_samples is not None and zs.size(0) > max_samples:
            zs = zs[:max_samples]
            break

    # Cov_p(x) [ E_q(z|x) [z_d] ]
    N = zs.size(0)
    cov_expected_z_given_x = 1 / (N - 1) * torch.sum((zs - zs.mean(0, keepdim=True)).pow(2), dim=0)

    # AU = \sum_d \delta{ Cov_p(x) [ E_q(z|x) [z_d] ] > \epsilon }
    au = (cov_expected_z_given_x > epsilon).float().sum()

    return Diagnostic({'active_units': {'au': au,
                                        'r_au': au / cov_expected_z_given_x.shape[0],
                                        'n': cov_expected_z_given_x.shape[0]}})
