from .estimators import *
from .model import PseudoCategorical
from .utils import *

_EPS = 1e-20


class OptCovReinforce(Reinforce):
    """
    Overleaf: https://www.overleaf.com/project/5d84f62f0c4fb30001e934ac

    c_\mathrm{opt}(x) = \frac{\sum_k \sum_m h^T_{mk} \sum_{m'} h_{m'k} f_{m'k} }{\sum_k  \sum_m h^T_{mk} \sum_{m'} h_{m'k} }

    # todo: one baseline for each N
    # todo: only leave sample km out
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_variate_loss_weight = 0  # the control variate doesn't have any parameter
        self.log_iw_m1 = np.log(self.iw - 1)
        self.detach_qlogits = True  # detach q since the reinforce loss only accounts for the parameters of phi, so the backward pass of L_k only deals with theta

    def compute_control_variate(self, x: Tensor, mc_estimate: bool = True, nz_estimate: bool = False,
                                **data: Dict[str, Tensor]) -> Tensor:
        """
        Compute the baseline that will be substracted to the score L_k,
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]
        :param x: input tensor
        :param mc_estimate: compute independent estimates for each MC sample
        :param nz_estimate: compute independent estimates for each latent variable
        :param data: additional data
        """
        bs, *dims = x.size()

        log_f_xz, z, qz = [data[k] for k in ["log_f_xz", "z", "qz"]]

        assert len(qz) == 1
        z, qz = z[0], qz[0]

        # convert to double
        qz.logits = qz.logits.double()
        z = z.double()
        log_f_xz = log_f_xz.double()

        # compute log probs
        qlogits = qz.logits
        log_qz = qz.log_prob(z)

        # d q(z|x) / d qlogits
        d_qlogits, = torch.autograd.grad(
            [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)

        # reshaping d_qlogits and qlogits
        N, K = d_qlogits.size()[1:]

        d_qlogits = d_qlogits.view(bs, self.mc, self.iw, N, K)

        with torch.no_grad():

            if nz_estimate:
                _n, _k = N, K
            else:
                _n, _k = 1, N * K

            # using notation from the overleaf doc with:
            # m = m
            # n = n
            h_kn = d_qlogits.view(bs, self.mc, self.iw, _n, _k)

            # define the operator \sum_{i != j}
            def sum_except(x, dim):
                # don't know why but this doesn't work
                return torch.sum(x, dim=dim, keepdim=True) - x

            def log_sum_exp_except(tensor, dim=-1, eps: float = 1e-20):
                max, _ = torch.max(tensor, dim=dim, keepdim=True)
                sum_exp = sum_except(torch.exp(tensor - max), dim)
                return torch.log(sum_exp + eps) + max

            log_w = log_f_xz.view(-1, self.mc, self.iw)

            debug = False
            if debug:
                log_w = log_w.detach()
                log_w.requires_grad = True

            # f_km = log_f_xz - v
            log_sum_exp_w = log_sum_exp_except(log_w, dim=2)
            f_km_no_m = log_sum_exp_w - self.log_iw_m1
            v_kmn = (log_w[:, :, :, None] - log_sum_exp_w[:, :, None, :]).exp()
            v_kmn = v_kmn.clamp(_EPS, 1 - _EPS)  # there may be a better way to solve numerical stability issues
            f_kmn_no_m = f_km_no_m[:, :, :, None] - v_kmn

            # flatten data so we can compute the sum_{m'k' != mk}
            h_kn = h_kn.view(bs, self.mc, self.iw, _n, -1)
            f_kmn_no_m = f_kmn_no_m[:, :, :, :, None, None].view(bs, self.mc, self.iw, self.iw, 1, 1)
            mask = 1 - torch.eye(self.iw, device=x.device, dtype=x.dtype)[None, None, :, :, None, None]

            # compute sums
            _h_kmn = h_kn[:, :, None, :, :, :].expand(bs, self.mc, self.iw, self.iw, _n, -1)
            s_hf = torch.sum(mask * _h_kmn * f_kmn_no_m, 3)
            s_h = sum_except(h_kn, 2)

            # dot product (over z_k dim)
            num = torch.sum(s_h * s_hf, -1)
            den = torch.sum(s_h * s_h, -1)

            if mc_estimate:
                # sum over mc samples
                num = num.sum(1, keepdim=True)
                den = den.sum(1, keepdim=True)

            # c_opt baseline
            c_opt = num / den

            # debugging : check for `nan` and unreasonable values for `v`
            if torch.sum(torch.isnan(c_opt)) or c_opt.mean().abs() > 1e3:
                print(
                    f" >> num = {num.mean().item():.3f}, den = {den.mean().item():.3f}, c_opt = {c_opt.mean().item():.3f}, v = {v_kmn.mean().item():.3E}")

                if debug:
                    (10 * c_opt[0, 0, 0] ** 2).mean().backward()
                    print(">>> grads: log_w:", log_w.grad[0])

            # if any nan, just replace with `0`
            c_opt[c_opt != c_opt] = 0

        return c_opt.detach().type(x.dtype)


class OptCovReinforce__(VariationalInference):
    """
    Overleaf: https://www.overleaf.com/project/5d84f62f0c4fb30001e934ac

    c_\mathrm{opt}(x) = \frac{\sum_k \sum_m h^T_{mk} \sum_{m'} h_{m'k} f_{m'k} }{\sum_k  \sum_m h^T_{mk} \sum_{m'} h_{m'k} }

    """

    def __init__(self, *args, baseline: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline is not None  # small hack to allow testing the estimator without the control variate.

    def f(self, model, qz, pz, x, z):
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = batch_reduce(log_qz - log_pz)
        elbo = log_px_z - kl
        return elbo, kl, -log_px_z, px

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, mc_baseline=True, id_estimates=False,
                **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
        bs, *dims = x.shape

        # expand sample
        x = self._expand_sample(x)

        qlogits = model.infer(x)
        qlogits.retain_grad()

        # p(z) and q(z|x)
        pz = PseudoCategorical(model.prior)
        qz = PseudoCategorical(qlogits)

        # z ~ q(z|x)
        z = qz.sample()

        # p(x|z)
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = batch_reduce(log_qz - log_pz)
        nll = - log_px_z

        # warning: detach kl such as L_k is only differentiable with regard to theta
        kl = kl.detach()

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(bs, self.mc, self.iw)

        # IW-ELBO: L_k = E_q(z_{1..K} | x) [ log 1/K \sum_{i=1..K} f(x, z_i)]
        L_k = torch.logsumexp(log_f_xz, dim=2) - self.log_iw if self.iw > 1 else log_f_xz.squeeze(2)

        # N_eff
        N_eff = self.effective_sample_size(log_pz, log_qz)

        # d q(z|x) / d qlogits
        d_qlogits = torch.autograd.grad(
            [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)[0]

        # reshaping d_qlogits and qlogits
        d_qlogits = d_qlogits.view(bs, self.mc, self.iw, *d_qlogits.size()[1:])
        qlogits = qlogits.view(bs, self.mc, self.iw, *qlogits.size()[1:])
        nll = nll.view(bs, self.mc, self.iw).mean(2)

        # vector control variate
        with torch.no_grad():
            v_mk = torch.exp(log_f_xz - torch.logsumexp(log_f_xz, dim=2, keepdim=True))

            # using notation from the overleaf doc
            h = d_qlogits.view(bs, self.mc, self.iw, -1)
            f_km = (L_k[:, :, None] - v_mk)

            # define the operator \sum_{i != j}
            sum_except = lambda x, dim: torch.sum(x, dim=dim, keepdim=True) - x

            if mc_baseline:
                # flatten data so we can compute the sum_{m'k' != mk}
                h = h.view(bs, self.mc * self.iw, -1)
                f_km = f_km.view(bs, self.mc * self.iw)

                # compute the optimal baseline
                s_h = sum_except(h, 1)
                s_hf = sum_except(h * f_km[:, :, None], 1)
                num = torch.sum(s_h * s_hf, -1)
                den = torch.sum(s_h * s_h, -1)

                c_opt = num / den
                c_opt = c_opt.view(bs, self.mc, self.iw)
                f_km = f_km.view(bs, self.mc, self.iw)

            else:
                # compute the estimate independently for ech mc sample
                s_h = sum_except(h, 2)
                s_hf = sum_except(h * f_km[:, :, :, None], 2)
                num = torch.sum(s_h * s_hf, -1)
                den = torch.sum(s_h * s_h, -1)

                c_opt = num / den

            # print(f">>> c_opt = {c_opt.shape}, ")

            # # compute Amm and bm
            # Amn = torch.einsum("bkmh, bknh -> bkmn", [h, h])
            # bm = torch.einsum("bkmh, bkh -> bkm", [h, torch.sum(h * f_mk[:,:,:,None], dim=2)])
            #
            # # MC averaging
            # if exclude_sample:
            #     Amn = (Amn.sum(1, keepdim=True) - Amn) / (self.mc-1)
            #     bm = (bm.sum(1, keepdim=True) - bm) / (self.mc-1)
            # else:
            #     Amn = Amn.mean(1, keepdim=True)
            #     bm = bm.mean(1, keepdim=True)
            #
            #
            # # scalar baseline
            # if scalar_baseline:
            #     Amn = Amn.sum((-2, -1,))
            #     bm = bm.sum((-1,))
            #     c_opt = bm / Amn
            # else:
            #     # compute Amn^{-1}
            #     shp = Amn.shape
            #     Amn = Amn.view(-1, *shp[-2:])
            #     Amn_inv = torch.pinverse(Amn, rcond=1e-18)  # use pseudo inverse for stability
            #     Amn_inv = Amn_inv.view(shp)
            #
            #     # baseline: C_opt = Amn^{-1} bm
            #     c_opt = (Amn_inv @ bm.unsqueeze(-1)).squeeze(-1)

            # reinforce score
            score = (f_km - c_opt)  # if self.baseline else f_mk

            control_variate_mse = (f_km - c_opt).pow(2).mean((1, 2))

        # reinforce loss
        __log_qz = log_qz.view(bs, self.mc, self.iw, -1)
        reinforce_loss = score[:, :, :, None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum((2, 3))  # sum over z and iw

        # MC averaging
        nll, L_k, reinforce_loss = map(lambda x: x.mean(1), (nll, L_k, reinforce_loss))

        # nll gives the gradients for theta
        loss = -L_k - reinforce_loss

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': nll,
                     'kl': kl,
                     'N_eff': N_eff,
                     'reinforce_loss': reinforce_loss,
                     'control_variate_mse': control_variate_mse}
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, {'x_': px, 'z': z, 'qz': qz, 'pz': pz, 'qlogits': qlogits}
