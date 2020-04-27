from torch import softmax

from .vi import *


class Reinforce(VariationalInference):
    """
    In the general case:

    L = E_q [ log w(x, z) ], w(x,z) = p(x,z)/q(z|x)
    d/dphi L = d/d phi \sum_z q(z|x) log w(x,z)
             = \sum_z log w(x,z) d/dphi q(z|x) + \sum_z q(z_x) d/dphi w(x,z)
             = \sum_z q(z|x) log w(x,z) d/dphi log q(z|x) + \sum_z q(z,x) d/dphi w(x,z)
             = E_q [ log w(x,z) d/dphi log q(z|x) + d/dphi w(x,z)] [ case (a) ]
             = E_q [ (log w(x,z) d/dphi - v) log q(z|x) ], v = 1  [ case (b) ]
    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.control_variate_loss_weight = 0 if baseline is None else 1.
        assert self.sequential_computation == False

        # `factorize_v` == True results in using case (b)
        self.factorize_v = False

    def compute_control_variate(self, x: Tensor, **data: Dict[str, Tensor]) -> Tuple[Tensor, dict, int]:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains `kwargs` and the outputs of the methods `compute_iw_bound` and `evaluate_model`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        if self.baseline is None:
            return torch.zeros((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype), {}, 0

        baseline = self.baseline(x)
        n_nans = len(baseline[baseline != baseline])
        meta = {}

        return baseline.view((x.size(0), 1, 1, 1)), meta, n_nans  # output of shape [bs, 1, 1, 1], Number of NaNs

    def compute_control_variate_l1(self, score, control_variate, weights=None):
        """L1 between the score function and its estimate"""

        if weights is None:
            weights = torch.ones_like(score)

        diff = (control_variate - score[:, :, :, None].detach())
        return (weights[..., None] * diff).abs().mean(3)

    def compute_reinforce_loss(self, score, control_variate, log_qz):
        log_qz = log_qz.view(score.size(0), self.mc, self.iw, -1)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        reinforce_loss = (score[:, :, :, None] - control_variate).detach() * log_qz

        # sum over iw: log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        return reinforce_loss.sum((2, 3))  # sum over z and iw

    def normalized_importance_weights(self, log_wk):
        return softmax(log_wk, dim=2)

    def compute_score(self, iw_data):
        L_k, kl, log_wk = [iw_data[k] for k in ('L_k', 'kl', 'log_wk')]
        v = self.normalized_importance_weights(log_wk)

        if self.factorize_v:
            score = L_k[:, :, None] - v
        else:
            score = L_k[:, :, None]

        return score, {'L_k': L_k, "v": v}

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, debug: bool = False, beta=1.0,
                **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:

        bs = x.size(0)

        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=self.factorize_v, beta=beta)
        L_k, kl, N_eff = [iw_data[k] for k in ('L_k', 'kl', 'N_eff')]
        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1)

        # compute score l: dL(z)\dtheta = L dq(z) \ dtheta
        score, score_meta = self.compute_score(iw_data)

        # compute control variate and MSE
        control_variate, control_variate_meta, _n_nans = self.compute_control_variate(x, **iw_data, **output, **kwargs)
        control_variate_l1 = self.compute_control_variate_l1(score, control_variate).mean((1, 2,))

        # reinforce loss
        reinforce_loss = self.compute_reinforce_loss(score, control_variate, log_qz)

        # MC averaging
        reinforce_loss = reinforce_loss.mean(1)
        L_k = L_k.mean(1)

        # final loss
        loss = - L_k - reinforce_loss + self.control_variate_loss_weight * control_variate_l1

        # update output with meta data
        output.update(**score_meta)
        output.update(**control_variate_meta)

        # compute dlogits
        if debug:
            output['dqlogits'] = self._compute_dlogits(output)
            output.update(**iw_data)

        # check shapes
        assert score.shape[0] == bs
        assert score.shape[1] == self.mc

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k,
                'nll': - self._reduce_sample(log_px_z),
                'kl': self._reduce_sample(kl),
                'r_eff': N_eff / self.iw
            },
            'reinforce': {
                'loss': reinforce_loss,
                'l1': control_variate_l1,
                'NaNs': _n_nans,
            },
        })

        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output

    def _compute_dlogits(self, output):

        z, qz = [output[k] for k in ['z', 'qz']]

        assert len(qz) == 1
        z, qz = z[0], qz[0]

        # compute log probs
        qlogits = qz.logits
        log_qz = qz.log_prob(z)

        # d q(z|x) / d qlogits
        d_qlogits, = torch.autograd.grad(
            [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)

        # reshaping d_qlogits and qlogits
        N, K = d_qlogits.size()[1:] if len(d_qlogits.shape) > 2 else (d_qlogits.size(1), 1)

        return d_qlogits.view(-1, self.mc, self.iw, N, K).detach()


class Vimco(Reinforce):
    """
    VIMCO: https://arxiv.org/abs/1602.06725
    Inspired by https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_variate_loss_weight = 0  # the control variate doesn't have any parameter
        self.log_iw_m1 = np.log(self.iw - 1)

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, arithmetic=False, return_raw=False,
                                use_outer_samples=False, use_double: bool = True, return_meta: bool = False,
                                **data: Dict[str, Tensor]) -> Tuple[
        Tensor, dict, int]:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains the output of the method `compute_iw_bound`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        log_wk = data['log_wk']
        log_wk = log_wk.view(-1, self.mc, self.iw)
        _dtype = log_wk.dtype

        if self.iw == 1:
            return torch.zeros_like(log_wk[:, :, 0]), {}, 0

        if use_double:
            log_wk = log_wk.double()

        if arithmetic:  # log \hat{f}(x, h^{-j}) using the arithmetic mean

            if use_outer_samples:

                mask = 1 - torch.eye(self.iw * self.mc, dtype=log_wk.dtype, device=log_wk.device)[None, :, :]
                _log_wk = log_wk[:, None, None, :, :].expand(-1, self.mc, self.iw, self.mc, self.iw)
                _log_wk = _log_wk.view(log_wk.size(0), self.mc * self.iw, self.mc * self.iw)

                # make sure to replace excluded samples with means so it doens't blow up to NAN with the exp (which gives `0` when multiplied by zero)
                _min, _ = _log_wk.min(dim=2, keepdim=True)
                _log_wk = (1 - mask) * _min + mask * _log_wk

                # compute the maximum for the log sum exp
                max, idx = _log_wk.max(dim=2, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_wk - max), dim=2)
                log_wk_hat = max.squeeze(2) + torch.log(
                    sum_exp) - self.log_mc_iw_m1  # adding eps should be necessary since the sum should be at least = exp(0)
                log_wk_hat = log_wk_hat.view(log_wk.size(0), self.mc, self.iw)

            else:
                mask = 1 - torch.eye(self.iw, dtype=log_wk.dtype, device=log_wk.device)[None, None, :, :]
                _log_wk = log_wk[:, :, None, :].expand(-1, self.mc, self.iw, self.iw)

                # make sure to replace excluded samples with means so it doens't blow up to NAN with the exp (which gives `0` when multiplied by zero)
                _min, _ = _log_wk.min(dim=3, keepdim=True)
                _log_wk = (1 - mask) * _min + mask * _log_wk

                # get the max for the LSE trick
                max, idx = _log_wk.max(dim=3, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_wk - max), dim=3)
                log_wk_hat = max.squeeze(3) + torch.log(sum_exp) - self.log_iw_m1

        else:  # log \hat{f}(x, h^{-j}) using the geometric mean
            if use_outer_samples:
                log_wk_hat = (torch.sum(log_wk, dim=(1, 2), keepdim=True) - log_wk) / (self.mc * self.iw - 1)
            else:
                log_wk_hat = (torch.sum(log_wk, dim=2, keepdim=True) - log_wk) / (self.iw - 1)

        log_wk_samples = log_wk.unsqueeze(-1) + torch.diag_embed(log_wk_hat - log_wk)
        baseline = torch.logsumexp(log_wk_samples, dim=2) - self.log_iw

        # catchning nans
        _n_nans = len(baseline[baseline != baseline])
        if _n_nans > 0:
            print(">>> vimco:compute_control_variate: baseline NAN : ", len(baseline[baseline != baseline]))
            if torch.isnan(baseline).all():
                baseline[baseline != baseline] = 0
            else:
                baseline[baseline != baseline] = baseline[baseline == baseline].mean()

        baseline = baseline.unsqueeze(-1)  # unsqueeze over Nz
        if return_meta:
            meta = {'L_hat': baseline}
        else:
            meta = {}

        if return_raw:
            return baseline, meta, _n_nans
        else:
            return baseline.type(_dtype), meta, _n_nans  # output of shape [bs, mc, iw, 1]


class VimcoPlus(Reinforce):
    """
    \nabla_phi L_K = E_q(z^1 .. z^K | x) [ \sum_k  (\eta * \log \gamma^K(v_k)) - \alpha v_k + v^{-k}) \nabla_phi log q(z^k | x) ]

    where:
    * w_k = p(x, z^k) / q(z^k | x)
    * v_k = w_k / \sum_l w_l
    * \alpha = 1 (ubiased) or \alpha < 1 (biased)
    * \eta = 1 : parameter to experiment with
    * \gamma^K(v_k) = (1 - 1/K) / (1 - v_k)
    * v^{-k} =
        * 0 : vimco
        * 1/K : copt/vimco++
    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.log_1_m_uniform = np.log(1. - 1. / self.iw)

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, debug: bool = False, alpha=1.0, eta=1.0,
                beta=1.0,
                v_k_hat='vimco', **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        bs = x.size(0)
        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=True, beta=beta)
        L_k, kl, N_eff, log_wk = [iw_data[k] for k in ('L_k', 'kl', 'N_eff', 'log_wk')]

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1).view(bs, self.mc, self.iw, -1)

        # compute v_k = w_k / \sum_l w_l
        v_k = self.normalized_importance_weights(log_wk)

        # compute \log \gamma^K(v_k)
        v_k_safe = torch.min((1 - 1e-6) * torch.ones_like(v_k), v_k)
        log_gamma_K = self.log_1_m_uniform - torch.log1p(- v_k_safe)

        v_k_hat = {'vimco': 0., 'copt': 1. / self.iw}[v_k_hat]

        # compute loss
        prefactor_k = (eta * log_gamma_K - alpha * v_k + v_k_hat)
        reinforce_loss = torch.sum(prefactor_k[..., None].detach() * log_qz, dim=(2, 3))  # sum over IW and Nz

        # MC averaging
        L_k = L_k.mean(1)
        reinforce_loss = reinforce_loss.mean(1)

        # final loss
        loss = - L_k - reinforce_loss

        # compute dlogits
        if debug:
            output['dqlogits'] = self._compute_dlogits(output)
            output.update(**iw_data)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k,
                'nll': - self._reduce_sample(log_px_z),
                'kl': self._reduce_sample(kl),
                'r_eff': N_eff / self.iw
            },
            'reinforce': {
                'loss': reinforce_loss,
            },
        })

        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
