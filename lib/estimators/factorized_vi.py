from .base import _EPS
from .reinforce import *


class FactorizedVariationalInference(Reinforce):
    """
    Variational Inference. Using this estimator requires the model to be compatible with the reparametrization trick.

    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def compute_partial_score(self, log_f_xz: Tensor):
        v_k = self.normalized_importance_weights(log_f_xz)
        return - v_k[..., None]

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> \
            Tuple[Tensor, Dict, Dict]:
        bs = x.size(0)

        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]

        # compute loss differentiable with regards to \theta
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=True)
        L_k_theta, kl, N_eff, log_f_xz = [iw_data[k] for k in ('L_k', 'kl', 'N_eff', 'log_f_xz')]

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1)

        # compute loss differentiable with regards to \phi
        log_qz = log_qz.view(bs, self.mc, self.iw, -1)
        # compute v_k
        partial_score = self.compute_partial_score(log_f_xz)
        # compute control variate c_k
        c_k, meta, _nans = self.compute_control_variate(x, **iw_data, **output)
        factor = .5 # todo: seems to work ok

        # \nabla_{\phi} L_k = E_{\eps^{1:K}} [ \sum_k v_k \nabla_{\phi} log q(z^k | x) ]
        L_k_phi = ((partial_score - factor * c_k).detach() * log_qz).sum(dim=(2, 3))  # sum over IW and z

        # compute loss, each part differentiable with regards to \theta and \phi
        loss = - (L_k_theta + L_k_phi)

        L_k_theta = L_k_theta.mean(1)
        loss = loss.mean(1)

        output.update(**meta)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k_theta,
                'nll': - self._reduce_sample(log_px_z),
                'kl': self._reduce_sample(kl),
                'r_eff': N_eff / self.iw
            },
            'prior': self.prior_diagnostics(output)
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output


class ControlledVariationalInference(FactorizedVariationalInference):

    def compute_control_variate(self, x: Tensor, mc_estimate: bool = False, arithmetic: bool = True,
                                use_outer_samples: bool = False,
                                nz_estimate: bool = False,
                                use_double: bool = True,
                                **data: Dict[str, Tensor]) -> Tuple[Tensor, dict, int]:
        """
        Compute the baseline that will be substracted to the score L_k,
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]
        :param x: input tensor
        :param mc_estimate: compute independent estimates for each MC sample
        :param vimco_estimate: estimate the current sample given the geometric average of the others: \hat{L}(z_m | z_{-m})
        :param nz_estimate: compute independent estimates for each latent variable
        :param data: additional data
        """
        bs, *dims = x.size()

        L_k, log_f_xz, z, qz = [data[k] for k in ["L_k", "log_f_xz", "z", "qz"]]

        if self.iw == 1:
            log_f_xz = log_f_xz.view(-1, self.mc, self.iw)
            return torch.zeros_like(log_f_xz[:, :, 0]), {}, 0

        assert len(qz) == 1
        z, qz = z[0], qz[0]

        # convert to double
        if use_double:
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
        N, K = d_qlogits.size()[1:] if len(d_qlogits.shape) > 2 else (d_qlogits.size(1), 1)

        d_qlogits = d_qlogits.view(bs, self.mc, self.iw, N, K)

        with torch.no_grad():

            if nz_estimate:
                _n, _k = N, K
            else:
                _n, _k = 1, N * K

            # using notation from the overleaf
            h_m = d_qlogits.view(bs, self.mc, self.iw, _n, _k)

            # define the operator \sum_{i != j}
            def sum_except(x, dim):
                # don't know why but this doesn't work
                return torch.sum(x, dim=dim, keepdim=True) - x

            def log_sum_exp_except(tensor, dim=-1, eps: float = _EPS, max=None):
                if max is None:
                    max, _ = torch.max(tensor, dim=dim, keepdim=True)
                sum_exp = sum_except(torch.exp(tensor - max), dim)
                return torch.log(eps + sum_exp) + max

            def log_sum_exp_except_stable(tensor, dim=-1, mask=None):
                assert tensor.shape == (bs, self.mc, self.iw)
                assert dim == 2

                # expand sample
                tensor = tensor[:, :, None, :].expand(bs, self.mc, self.iw, self.iw)

                # set the excluded sample as the min for numerical stability
                _min, _ = tensor.min(dim=3, keepdim=True)
                tensor = (1 - mask) * _min + mask * tensor

                # max for the LSE trick
                max, idx = tensor.max(dim=3, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(tensor - max), dim=3)

                return max.squeeze(3) + torch.log(sum_exp)

            # log ratios
            log_w = log_f_xz.view(-1, self.mc, self.iw)

            # msk
            mask = 1 - torch.eye(self.iw, device=x.device, dtype=x.dtype)

            # compute the `v` term
            log_sum_exp_w = log_sum_exp_except_stable(log_w, dim=2, mask=mask)

            log_sum_exp_w_nans = len(log_sum_exp_w[log_sum_exp_w != log_sum_exp_w])
            _n_nans = log_sum_exp_w_nans
            if log_sum_exp_w_nans > 0:

                if torch.isnan(log_sum_exp_w).all():
                    log_sum_exp_w[log_sum_exp_w != log_sum_exp_w] = 0
                else:
                    log_sum_exp_w[log_sum_exp_w != log_sum_exp_w] = log_sum_exp_w[log_sum_exp_w == log_sum_exp_w].mean()
                print(">>> c*: log sum exp NAN, N =", _n_nans)

            log_v_mn = log_w[:, :, :, None] - log_sum_exp_w[:, :, None, :]

            # mask `m` samples to avoid getting `inf` and then 'NaN'
            _min, _ = log_v_mn.min(dim=-1, keepdim=True)
            _masked_log_vmn = (1 - mask[None, None, :, :]) * _min + mask[None, None, :, :] * log_v_mn

            v_mn = _masked_log_vmn.exp()

            h_m = h_m.view(bs, self.mc, self.iw, _n, -1)
            v_mn = v_mn.view(bs, self.mc, self.iw, self.iw)

            # shape = [bs, k, m, m', _n, h]
            #       = [bs, k, m, n, u, h]
            # with notation: m = m, m' = n, m'' = p, m''' = q, nz = u, kz = h

            # computing the weights: alpha_mn
            # Denominator (k is ignored in this notation as we can simply sum over it at the end)
            spsq_h_p_T_h_q = torch.einsum("bkpuh, bkquh -> bku", [h_m, h_m])
            sp_h_p_T_h_m = torch.einsum("bkpuh, bkmuh -> bkmu", [h_m, h_m])
            h_m_T_h_m = torch.einsum("bkmuh, bkmuh -> bkmu", [h_m, h_m])
            den = spsq_h_p_T_h_q[:, :, None, :] - 2 * sp_h_p_T_h_m + h_m_T_h_m
            if mc_estimate:
                den = den.sum(1, keepdims=True)

            # Numerator
            sp_h_p_T_h_n = torch.einsum("bkpuh, bknuh -> bknu", [h_m, h_m])
            h_m_T_h_n = torch.einsum("bkmuh, bknuh -> bkmnu", [h_m, h_m])
            num = sp_h_p_T_h_n[:, :, None, :, :] - h_m_T_h_n

            # weights alpha_mn: sum_{n != m} alpha_mn = 1
            alpha_mn = num / (_EPS + den[:, :, :, None, :])

            f_mn = -v_mn

            # control variate
            v_hat = torch.einsum("mn, bkmnu, bkmn -> bkmu", [mask, alpha_mn, f_mn])

            if mc_estimate:
                v_hat = v_hat.sum(1, keepdims=True)

            c_opt_nans = len(v_hat[v_hat != v_hat])

            _n_nans += c_opt_nans
            if c_opt_nans:
                print(">>> c*: c_opt NAN, N = ", _n_nans)

            # if any NaN, just replace with `0`
            v_hat[v_hat != v_hat] = 0

        return v_hat.detach().type(x.dtype), {}, _n_nans
