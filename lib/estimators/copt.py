from lib.utils import *
from .base import _EPS
from .reinforce import *


class OptCovReinforce(Reinforce):
    """
    Overleaf: https://www.overleaf.com/project/5d84f62f0c4fb30001e934ac

    c_\mathrm{opt}(x) = \frac{\sum_k \sum_m h^T_{mk} \sum_{m'} h_{m'k} f_{m'k} }{\sum_k  \sum_m h^T_{mk} \sum_{m'} h_{m'k} }

    # todo: only leave sample km out
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_variate_loss_weight = 0  # the control variate doesn't have any parameter
        self.log_iw_m1 = np.log(self.iw - 1)
        self.factorize_v = True

    def normalized_importance_weights(self, log_f_xz):
        v = softmax(log_f_xz, dim=2)
        if v[v > 1 + _EPS].sum() > 0:
            print(f"~~~~ warning | in:normalized_importance_weights: v> 1 {v[v > 1 + _EPS]}")
        return v

    def catch_error(self, v_kmn):

        # catching numerical errors

        # in some cases the terms v were > 1, this shouldn't happen
        if len(v_kmn[v_kmn.abs() > 1 + _EPS]) > 0:
            # raise ValueError("v_kmn > 1")

            print(">>> warning: v>1:", v_kmn[v_kmn.abs() > 1])
            # then make sure that \sum_n v_{*,n} = 1
            v_kmn = v_kmn / (_EPS + v_kmn.sum(-1, keepdims=True))

        # debugging
        if len(v_kmn[v_kmn.abs() > 1 + _EPS]) > 0:
            print(">>>> error: values of v_kmn are >1: v summary:", v_kmn.max(), v_kmn.min(), v_kmn.mean(),
                  v_kmn.std())
            print(">>>> v[v>1]", v_kmn[v_kmn.abs() > 1])

        return v_kmn

    def compute_control_variate(self, x: Tensor, mc_estimate: bool = False, arithmetic: bool = True,
                                use_outer_samples: bool = False,
                                nz_estimate: bool = False,
                                use_double: bool = True,
                                **data: Dict[str, Tensor]) -> Tensor:
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

        v = self.normalized_importance_weights(log_f_xz)
        score = L_k[:, :, None] - v

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
        N, K = d_qlogits.shape[1:] if len(d_qlogits.shape) > 2 else  d_qlogits.shape[1], 1

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

            def __nan_block(x, key):
                """raise value error if a `NaN` is detected"""
                n_infs = len(x[x.abs() == float('inf')])
                if n_infs:
                    print("# inf values:", n_infs)
                if len(x[x != x]) > 0:
                    raise ValueError(f"# NANS found in the computation of {key}, and that shouldn't happen")

            # log ratios
            log_w = log_f_xz.view(-1, self.mc, self.iw)

            # msk
            mask = 1 - torch.eye(self.iw, device=x.device, dtype=x.dtype)

            # compute \hat{L} \approx \log 1\k \sum_m w_m
            L_hat, _, _n_nans = Vimco.compute_control_variate(self, x, mc_estimate=mc_estimate, arithmetic=arithmetic,
                                                              return_raw=True, use_double=use_double,
                                                              use_outer_samples=use_outer_samples, **data)

            # compute the `v` term
            log_sum_exp_w = log_sum_exp_except_stable(log_w, dim=2, mask=mask)

            log_sum_exp_w_nans = len(log_sum_exp_w[log_sum_exp_w != log_sum_exp_w])
            _n_nans += log_sum_exp_w_nans
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

            # check if v_m < 1
            # todo: check if it sums to one
            v_mn = self.catch_error(mask[None, None, :, :] * v_mn)

            h_m = h_m.view(bs, self.mc, self.iw, _n, -1)
            v_mn = v_mn.view(bs, self.mc, self.iw, self.iw)
            L_hat = L_hat[:, :, :, None].view(bs, -1, self.iw, 1)

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

            # f_{k,m'}^{-m}
            f_mn = L_hat - v_mn

            # __nan_block(L_hat, "L_hat")
            # __nan_block(v_mn, "v_mn")
            # __nan_block(f_mn, "f_mn")

            # control variate
            c_opt = torch.einsum("mn, bkmnu, bkmn -> bkmu", [mask, alpha_mn, f_mn])

            if mc_estimate:
                c_opt = c_opt.sum(1, keepdims=True)

            c_opt_nans = len(c_opt[c_opt != c_opt])
            _n_nans += c_opt_nans
            if c_opt_nans:
                print(">>> c*: c_opt NAN, N = ", _n_nans)

            # if any NaN, just replace with `0`
            c_opt[c_opt != c_opt] = 0

        L_hat = torch.einsum("mn, bkmnu, bkmn -> bkmu", [mask, alpha_mn, L_hat.expand_as(v_mn)])
        v_hat = torch.einsum("mn, bkmnu, bkmn -> bkmu", [mask, alpha_mn, v_mn])
        meta = {'L_hat': L_hat, 'v_hat': v_hat}

        return c_opt.detach().type(x.dtype), meta, _n_nans
