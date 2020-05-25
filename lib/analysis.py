import torch


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
