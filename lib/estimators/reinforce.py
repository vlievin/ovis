from .vi import *
from .base import _EPS


class ZScore(nn.Module):
    """
    Module used to filter individual samples in reinforce [highly experimental]
    """

    def __init__(self, momentum=0.001, eps=1e-05):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.initialized = False
        self.register_buffer("mean", torch.tensor(0.))
        self.register_buffer("variance", torch.tensor(1.))

    def update_statistics(self, x, momentum):
        self.mean = (1 - momentum) * self.mean + momentum * x.mean()
        self.variance = (1 - momentum) * self.variance + momentum * x.var()

    @torch.no_grad()
    def forward(self, x):
        z = (x - self.mean) / (self.eps + self.variance.sqrt())

        if self.training:

            if self.initialized:
                self.update_statistics(x, self.momentum)
            else:
                self.initialized = True
                self.update_statistics(x, 1)

        return z

    @torch.no_grad()
    def get_threshold(self, z_reject):
        return z_reject * (self.eps + self.variance.sqrt()) + self.mean


class Reinforce(VariationalInference):
    """
    Reinforce with optional baseline:

    * (a) in the general case:

    L = E_q [ log f(x, z) ], f(x,z) = p(x,z)/q(z|x)
    d/d theta L = d/d theta \sum_z q(z|x) log f(x,z)
             = \sum_z log f(x,z) d/d theta q(z|x) + \sum_z q(z_x) d/d theta f(x,z)
             = \sum_z q(z|x) log f(x,z) d/d theta log q(z|x) + \sum_z q(z,x) d/d theta f(x,z)
             = E_q [ log f(x,z) d/d theta log q(z|x) + d/d theta f(x,z)]


    * (b) when deriving from `phi`, the parameters of the posterior, in that case L_k will be only differentiated with regards to `theta`
    d/d_phi L_k =  E_q [ [ log( 1/K \sum_k w(x,z_k) ) - v_k ] d/d_phi log q(z|x), where v_k = w_m / \sum_m w_m

    in that case we define the score as E_q[ log( 1/K \sum_k w(x,z_k) ) - v_k ]

    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.control_variate_loss_weight = 0 if baseline is None else 1.
        assert self.sequential_computation == False

        # `factorize_v` == True results in using case (b)
        self.factorize_v = False

        # measure distribution of L1 for rejection sampling
        self.z_score_l1 = ZScore()

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

    def compute_reinforce_loss(self, score, control_variate, log_qz, weights=None):

        log_qz = log_qz.view(score.size(0), self.mc, self.iw, -1)

        if weights is None:
            weights = torch.ones_like(score)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        reinforce_loss = (score[:, :, :, None] - control_variate).detach() * log_qz

        # sum over iw: log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        return (weights[..., None] * reinforce_loss).sum((2, 3))  # sum over z and iw

    def normalized_importance_weights(self, log_f_xz):
        v = softmax(log_f_xz, dim=2)
        if v[v > 1 + _EPS].sum() > 0:
            print(f"~~~~ warning | in:normalized_importance_weights: v> 1 {v[v > 1 + _EPS]}")
        return v

    def compute_score(self, iw_data, mc_estimate):
        L_k, kl, log_f_xz = [iw_data[k] for k in ('L_k', 'kl', 'log_f_xz')]
        v = self.normalized_importance_weights(log_f_xz)

        if self.factorize_v:
            score = L_k[:, :, None] - v
        else:
            score = L_k[:, :, None]

        if mc_estimate:
            score = score.mean(1, keepdim=True)

        return score, {'L_k': L_k, "v": v}

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, mc_estimate: bool = False,
                factorize_v: bool = None, z_reject=0,
                **kwargs: Any) -> \
            Tuple[Tensor, Dict, Dict]:

        bs = x.size(0)

        # todo: hacky change of state, implement a clean version
        if factorize_v is not None:
            self.factorize_v = factorize_v

        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=self.factorize_v)
        L_k, kl, N_eff = [iw_data[k] for k in ('L_k', 'kl', 'N_eff')]

        # compute score l: dL(z)\dtheta = L dq(z) \ dtheta
        score, score_meta = self.compute_score(iw_data, mc_estimate=mc_estimate)

        # compute control variate and MSE
        control_variate, control_variate_meta, _n_nans = self.compute_control_variate(x, mc_estimate=mc_estimate,
                                                                                      **iw_data, **output,
                                                                                      **kwargs)
        control_variate_l1 = self.compute_control_variate_l1(score, control_variate)

        # rejection sampling according to the control variate L1
        if z_reject > 0:
            z_score_l1 = self.z_score_l1(control_variate_l1)
            reject_weights = (z_score_l1 < z_reject).float()

            reject_ratio = (1 - reject_weights).sum() / reject_weights.view(-1).shape[0]
            l1_threshold = self.z_score_l1.get_threshold(z_reject)

            if reject_ratio > 0.5 or (not self.z_score_l1.initialized):  # safety
                reject_weights = None
                reject_ratio = 0
        else:
            reject_weights = None
            reject_ratio = 0.
            l1_threshold = 0.

        # log filtered f_m to a file for debugging
        # if reject_weights is not None:
        #     with torch.no_grad():
        #         log_f_xz = iw_data.get('log_f_xz')
        #         for b, w_b in enumerate(reject_weights):
        #             for k, w_k in enumerate(w_b):
        #                 if w_k.sum() / len(w_k.view(-1)) < 1:
        #                     f_s = log_f_xz[b, k]
        #                     f_keep = f_s[w_k==1]
        #                     f_reject = f_s[w_k == 0]
        #                     p_f_keep = Normal(f_keep.mean(), f_keep.std())
        #                     print(f">> p(reject | keep) {p_f_keep.log_prob(f_reject).exp().mean().item():.2E}     ,p(keep | keep) {p_f_keep.log_prob(f_keep).exp().mean().item():.2E}")

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1)

        # reinforce loss
        reinforce_loss = self.compute_reinforce_loss(score, control_variate, log_qz, weights=reject_weights)

        # averaging MSE
        control_variate_l1_raw = control_variate_l1.mean((1, 2,))
        if reject_weights is None:
            control_variate_l1 = control_variate_l1_raw
        else:
            control_variate_l1 = self.compute_control_variate_l1(score, control_variate, weights=reject_weights)
            control_variate_l1 = control_variate_l1.sum(dim=(1, 2,)) / reject_weights.sum(dim=(1, 2,))

        # MC averaging
        reinforce_loss = reinforce_loss.mean(1)
        if reject_weights is not None:
            # the reinforce term (score - baseline) * dlogits is filtered using the rejection rule
            # however the L_k terms still contains all IW samples information
            # one intuitive solution is to weight each L_k term by the number of `active` samples.

            w_norm = 'exp'

            m_weights = reject_weights.sum(2)

            if w_norm == 'exp':
                w = m_weights / reject_weights.shape[2]
            else:
                w = torch.exp(m_weights - reject_weights.shape[2])

            w = w / w.sum(1, keepdim=True)
            _L_k = (w * L_k).sum(1)
            L_k = L_k.mean(1)
        else:
            _L_k = L_k = L_k.mean(1)

        # final loss
        loss = - _L_k - reinforce_loss + self.control_variate_loss_weight * control_variate_l1

        # update output with meta data
        output.update(**score_meta)
        output.update(**control_variate_meta)

        # compute dlogits
        output['dqlogits'] = self._compute_dlogits(output)

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
                'l1_raw': control_variate_l1_raw,
                'l1_threshold': l1_threshold,
                'NaNs': _n_nans,
                'rejected': reject_ratio
            },
            'prior': self.prior_diagnostics(output)
        })

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
        N, K = d_qlogits.shape[1:] if len(d_qlogits.shape) > 2 else  d_qlogits.shape[1], 1

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
    def compute_control_variate(self, x: Tensor, mc_estimate: bool = True, arithmetic=False, return_raw=False,
                                use_outer_samples=False, use_double: bool = True, **data: Dict[str, Tensor]) -> Tensor:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains the output of the method `compute_iw_bound`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        log_f_xz = data['log_f_xz']
        log_f_xz = log_f_xz.view(-1, self.mc, self.iw)
        _dtype = log_f_xz.dtype
        if use_double:
            log_f_xz = log_f_xz.double()

        if arithmetic:  # log \hat{f}(x, h^{-j}) using the arithmetic mean

            if use_outer_samples:

                mask = 1 - torch.eye(self.iw * self.mc, dtype=log_f_xz.dtype, device=log_f_xz.device)[None, :, :]
                _log_f_xz = log_f_xz[:, None, None, :, :].expand(-1, self.mc, self.iw, self.mc, self.iw)
                _log_f_xz = _log_f_xz.view(log_f_xz.size(0), self.mc * self.iw, self.mc * self.iw)

                # make sure to replace excluded samples with means so it doens't blow up to NAN with the exp (which gives `0` when multiplied by zero)
                _min, _ = _log_f_xz.min(dim=2, keepdim=True)
                _log_f_xz = (1 - mask) * _min + mask * _log_f_xz

                # compute the maximum for the log sum exp
                max, idx = _log_f_xz.max(dim=2, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_f_xz - max), dim=2)
                log_f_xz_hat = max.squeeze(2) + torch.log(
                    sum_exp) - self.log_mc_iw_m1  # adding eps should be necessary since the sum should be at least = exp(0)
                log_f_xz_hat = log_f_xz_hat.view(log_f_xz.size(0), self.mc, self.iw)

            else:
                mask = 1 - torch.eye(self.iw, dtype=log_f_xz.dtype, device=log_f_xz.device)[None, None, :, :]
                _log_f_xz = log_f_xz[:, :, None, :].expand(-1, self.mc, self.iw, self.iw)

                # make sure to replace excluded samples with means so it doens't blow up to NAN with the exp (which gives `0` when multiplied by zero)
                _min, _ = _log_f_xz.min(dim=3, keepdim=True)
                _log_f_xz = (1 - mask) * _min + mask * _log_f_xz

                # get the max for the LSE trick
                max, idx = _log_f_xz.max(dim=3, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_f_xz - max), dim=3)
                log_f_xz_hat = max.squeeze(3) + torch.log(sum_exp) - self.log_iw_m1

        else:  # log \hat{f}(x, h^{-j}) using the geometric mean

            if use_outer_samples:
                log_f_xz_hat = (torch.sum(log_f_xz, dim=(1, 2), keepdim=True) - log_f_xz) / (self.mc * self.iw - 1)
            else:
                log_f_xz_hat = (torch.sum(log_f_xz, dim=2, keepdim=True) - log_f_xz) / (self.iw - 1)

        log_f_xz_samples = log_f_xz.unsqueeze(-1) + torch.diag_embed(log_f_xz_hat - log_f_xz)
        baseline = torch.logsumexp(log_f_xz_samples, dim=2) - self.log_iw

        # catchning nans

        _n_nans = len(baseline[baseline != baseline])
        if _n_nans > 0:

            print(">>> vimco:compute_control_variate: baseline NAN : ", len(baseline[baseline != baseline]))
            # print("scripts args = ", sys.argv)

            if torch.isnan(baseline).all():
                baseline[baseline != baseline] = 0
            else:
                baseline[baseline != baseline] = baseline[baseline == baseline].mean()

        if mc_estimate:
            baseline = baseline.mean(1, keepdim=True)

        baseline = baseline.unsqueeze(-1)  # unsqueeze over Nz

        meta = {'L_hat': baseline}

        if return_raw:
            return baseline, meta, _n_nans
        else:
            return baseline.type(_dtype), meta, _n_nans  # output of shape [bs, mc, iw, 1]
