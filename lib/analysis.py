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


def total_derivatives_analysis(estimator, model, x, mc_samples, **config):
    # expand x to `mc_samples` times
    x = x[0][None].repeat(mc_samples, *(1 for _ in x.shape[1:]))

    # forward pass
    _, _, output = estimator(model, x, analysis=True, **config)

    # unpack data
    gh, vh, gvh, gmuvh = [output[k].view(mc_samples, estimator.mc, estimator.iw, -1) for k in
                          ['gh', 'vh', 'gvh', 'gmuvh']]

    # compute estimates h = 1/M \sum_m \sum_k h_mk
    def estimate(h):
        return h.mean(1).sum(2)

    gh, vh, gvh, gmuvh = map(estimate, (gh, vh, gvh, gmuvh))

    # compute covariance matrices
    @torch.no_grad()
    def covariance(x, y=None):
        assert len(x.shape) == 2

        # center x and y
        x -= x.mean(0, keepdim=True)
        if y is None:
            y = x
        else:
            assert y.shape == x.shape
            y -= y.mean(0, keepdim=True)

        # get number of samples
        N = x.shape[0]
        assert y.shape[0] == y.shape[0]

        # conpute covariance
        return 1. / (N - 1) * x.transpose(1, 0) @ x

    cov_gh, cov_vh, cov_gvh, cov_gmuvh = map(covariance, (gh, vh, gvh, gmuvh))

    # total variance
    @torch.no_grad()
    def total_variance(cov):
        return cov.trace()

    var_gh, var_vh, var_gvh, var_gmuvh = map(total_variance, (cov_gh, cov_vh, cov_gvh, cov_gmuvh))

    print("Total Variance:")
    print(f"var_gamma_hk = {var_gh.item():.3E}")
    print(f"var_vk_hk = {var_vh.item():.3E}")
    print(f"var_g_vimco = {var_gvh.item():.3E}")
    print(f"var_g_copt = {var_gmuvh.item():.3E}")

    return {
        'total_variance': {"var_gamma_hk": var_gh, "var_vk": var_gh, "var_g_vimco": var_gvh, "var_g_copt": var_gmuvh}}
