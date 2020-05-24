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
        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=self.factorize_v, beta=beta)

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
        L_k = iw_data['L_k'].mean(1)

        # final loss
        loss = - L_k - reinforce_loss + self.control_variate_loss_weight * control_variate_l1

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
                **self._loss_diagnostics(**iw_data, **output)
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

        output.update(**score_meta)
        output.update(**control_variate_meta)
        return loss, diagnostics, output

    def _compute_dlogits_l(self, z, qz):

        # compute log probs
        qlogits = qz.logits
        log_qz = qz.log_prob(z)

        # d q(z|x) / d qlogits
        d_qlogits, = torch.autograd.grad(
            [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)

        # reshaping d_qlogits and qlogits
        N, K = d_qlogits.size()[1:] if len(d_qlogits.shape) > 2 else (d_qlogits.size(1), 1)

        return d_qlogits.view(-1, self.mc, self.iw, N * K).detach()

    def _compute_dlogits(self, output):

        z, qz = [output[k] for k in ['z', 'qz']]

        return torch.cat([self._compute_dlogits_l(z_l, qz_l) for z_l, qz_l in zip(z, qz)], -1)


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
                                use_outer_samples=False, use_double: bool = False, return_meta: bool = False,
                                **data: Dict[str, Tensor]) -> Tuple[
        Tensor, dict, int]:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains the output of the method `compute_iw_bound`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        log_wk = data['log_wk']
        log_wk = log_wk.view(-1, self.mc, self.iw)
        _dtype = log_wk.dtype
        log_mc_iw_m1 = np.log(self.iw - 1)

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
                    sum_exp) - log_mc_iw_m1  # adding eps should not be necessary since the sum should be at least = exp(0)
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
                log_wk_hat = max.squeeze(3) + torch.log(sum_exp) - log_mc_iw_m1

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
    \nabla_phi L_K = E_q(z^1 .. z^K | x) [ \sum_k  (\log Z_{1:K} - v_k - c_k) \nabla_phi log q(z^k | x) ]
    """

    def __init__(self, baseline: Optional[nn.Module] = None, auxiliary_samples=0, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.log_1_m_uniform = np.log(1. - 1. / self.iw)

        if auxiliary_samples > 0:
            self.iw += auxiliary_samples


        print(">>> VimcoPlus:", self.iw, auxiliary_samples)

    @torch.no_grad()
    def compute_prefactors(self,
                           L_k: Tensor,
                           log_wk: Tensor,
                           ess: Tensor,
                           hk: Optional[Tensor] = None,
                           mode: str = 'vimco',
                           truncation: float = 0,
                           alpha: float = 1.0,
                           autoalpha: bool = False,
                           alpha_mu: float = 2.5,
                           alpha_sigma: float = 1.0,
                           handle_low_ess: bool = False,
                           use_second_largest: bool = False,
                           center: bool = False,
                           biased: bool = False,
                           analysis: bool = False,
                           auxiliary_samples: int = 0,
                           **kwargs):

        if autoalpha:
            alpha = (((ess - alpha_mu) / alpha_sigma).sigmoid())[:, :, None]

        if auxiliary_samples > 0:
            log_wk_aux = log_wk[:, :, self.iw-auxiliary_samples:]
            log_wk = log_wk[:, :, :self.iw-auxiliary_samples]

        # compute v_k = w_k / \sum_l w_l
        v_k = self.normalized_importance_weights(log_wk)

        if truncation > 0:
            v_k = torch.max(v_k, truncation * torch.ones_like(v_k))

        v_k_safe = torch.min((1 - 1e-6) * torch.ones_like(v_k), v_k)

        # compute prefactors g_k such that g = \sum_k g_k h_k
        if mode == 'vimco-geometric':
            # c_k = log Z^{-k} = 1/K \sum{ l \neq k} w_l where log w_k = 1\(K-1) \sum {l \neq k} log w_l
            c_k, *_ = Vimco.compute_control_variate(self, None, arithmetic=False, log_wk=log_wk)
            gk = L_k[:, :, None] - c_k.sum(-1) - alpha * v_k
        elif mode == 'vimco':
            # c_k = log Z^{-k} = 1/K \sum{ l \neq k} w_l where w_k = 1\(K-1) \sum {l \neq k} w_l
            # g_k = log Z - c_k - v_k = log (1 - 1/K) - log(1 - v_k) - v_k
            gk = self.log_1_m_uniform - torch.log1p(- v_k_safe) - alpha * v_k
        elif mode == 'copt-uniform':
            # c_k = log Z^{-k} - 1/ K
            # g_k = log Z - c_k = log (1 - 1/K) - log(1 - v_k) + 1 / K - v_k
            gk = self.log_1_m_uniform - torch.log1p(- v_k_safe) + alpha * (1 / self.iw - v_k)
        elif mode == 'copt':
            # c_k = log 1/ K \sum_{l \neq k} w_l
            # g_k = log Z - c_k = - log(1-v_k) - v_k
            if auxiliary_samples == 0:
                gk = - torch.log1p(- v_k_safe) - alpha * v_k

            else:
                gk =  L_k[:, :, None] - alpha * v_k

                # remove the auxiliary samples
                aux = auxiliary_samples
                iw = self.iw - aux
                log_wk_ = log_wk.view(-1, self.mc, 1, 1, iw).expand(-1, self.mc, aux, iw, iw)
                log_wk_aux = log_wk_aux.view(-1, self.mc, aux, 1, 1).expand(-1, self.mc, aux, iw, iw)
                mask = 1 - torch.eye(iw, device=log_wk.device)
                mask = mask.view(1, 1, 1, iw, iw)
                log_wk_unbiased = mask * log_wk_ + (1-mask) * log_wk_aux

                vk_ = log_wk_unbiased.softmax(-1)
                L_k_= torch.logsumexp(log_wk_unbiased, dim=-1) - np.log(log_wk_unbiased.size(-1))

                gk_ = L_k_[:,:,:,:,None] - alpha * vk_

                expected_gk = gk_.diagonal(dim1=3, dim2=4).mean(2)


                # print(f">>> gk         : {gk.mean():.3f} [{gk.std():.3f} ]")
                # print(f">>> expected_gk: {expected_gk.mean():.3f} [{expected_gk.std():.3f} ]")
                #
                #
                # print(v_k[0,0,:])
                # print(gk[0,0,:])
                # print(expected_gk[0,0,:])
                # print(
                #     f">> gk = {gk.mean().item():.3E} (+/- {gk.std(dim=2).mean().item():.3E} ) range [{gk.min().item():.3E}-{gk.max().item():.3E}]")
                # print(
                #     f">> expected_gk = {expected_gk.mean().item():.3E} (+/- {expected_gk.std(dim=2).mean().item():.3E} )  range [{gk.min().item():.3E}-{gk.max().item():.3E}]")

                gk = gk - expected_gk

                # print(
                #     f">> gk - E[g_k] = {gk.mean().item():.3f} (+/- {gk.std(dim=2).mean().item():.3f} ) range [{gk.min().item():.3f}-{gk.max().item():.3f}]")




        elif mode == 'ww':
            # wake-wake estimator
            # g_k = v_k
            gk = v_k
        else:
            raise ValueError(f"Unknown mode VimcoPlus `mode` parameter `{mode}`")

        if handle_low_ess:

            # when ESS \approx 1, exploits that w_k >> \sum_{l \neq k} w_l
            # and use c_k = log Z{-k} - 1_{k = argmax w_k} (1+logK)
            if not use_second_largest:
                # mask = 1_{k = argmax w_k}
                log_wk = log_wk.view(-1, self.mc, self.iw)
                _max, idx = log_wk.max(dim=2, keepdim=True)
                mask = (log_wk == _max).float()

                # c_k = log Z-k - 1_{k = argmax w_k}
                # log Z - c_k - v_k = log (1 - 1/K) - log(1 - v_k) + \delta_{k = argmax w_k} - v_k
                _gk = gk + alpha * mask
            else:
                # todo: update to match with above version
                # use the second largest w_k instead of the average log Z^{-k}

                # mask = 1_{k = argmax w_k}
                log_wk = log_wk.view(-1, self.mc, self.iw)
                _topk, idx = log_wk.topk(k=2, dim=2)
                _max = _topk[:, :, 0][..., None]
                _second_max = _topk[:, :, 1][..., None]
                mask = (log_wk == _max).float()

                # c_k = {log w_i}_{i \neq k}.max() - 1_{k = argmax w_k}
                # log Z - c_k - v_k = L_k - {log w_i}_{i \neq k}.max() + 1_{k = argmax w_k} - v_k
                _gk = (1 - mask) * gk + mask * (
                        L_k[:, :, None] - _second_max + self.log_iw + alpha * (1 - v_k))

            # use the low ESS estimate only when ess \approx 1
            ess = ess[:, :, None].expand(-1, self.mc, self.iw)
            gk = torch.where(ess < 1.05, _gk, gk)

        # centering
        if center:
            assert hk is not None

            hk = hk.view(*gk.shape, -1)
            # hk_norm2 = (hk ** 2).sum(-1)
            q = hk.size(-1)
            eps = 1e-12
            mask = 1 - torch.eye(self.iw, device=hk.device).view(1, 1, self.iw, self.iw)

            # replicate
            fl = gk.view(-1, self.mc, 1, self.iw).expand(-1, self.mc, self.iw, self.iw)

            # hl = hk.view(-1, self.mc, self.iw, 1, q).expand(-1, self.mc, self.iw, self.iw, q)
            # hkl = hk.view(-1, self.mc, 1, self.iw, q).expand(-1, self.mc, self.iw, self.iw, q)
            # expected_gk = (mask * fl *  (hkl * hl).sum(-1)).sum(-1) / (eps + (mask * hk_norm2).sum(-1))
            expected_gk = (mask * fl).sum(-1) / mask.sum(-1)

            # h_w = (hl * hkl).sum(-1) / hkl.pow(2).sum(-1)
            #
            # h_w = h_w[mask > 0]

            # print(f"ess = {ess.mean().item():.3E}, K = {self.iw}, 1/K = {1/self.iw:.3E}, 1/K**0.5 = {1/self.iw**0.5:.3E}")
            # # print(
            # #     f">> h_w = {h_w.mean().item():.3E} (+/- {h_w.std().item():.3E} ) range [{h_w.min().item():.3E}-{h_w.max().item():.3E}]")
            # print(f">> gk = {gk.mean().item():.3E} (+/- {gk.std(dim=2).mean().item():.3E} ) range [{gk.min().item():.3E}-{gk.max().item():.3E}]")
            # print(f">> expected_gk = {expected_gk.mean().item():.3E} (+/- {expected_gk.std(dim=2).mean().item():.3E} )  range [{gk.min().item():.3E}-{gk.max().item():.3E}]")

            gk = gk - expected_gk

            # print(
            #     f">> gk - E[g_k] = {gk.mean().item():.3f} (+/- {gk.std(dim=2).mean().item():.3f} ) range [{gk.min().item():.3f}-{gk.max().item():.3f}]")

        diagnostics = dict()

        # hk_norm = hk.norm(dim=-1).pow(2).view(-1)
        # print(f">>> hk_norm = {hk_norm.mean().item():.3f} [{hk_norm.std().item():.3f}]")

        if analysis:
            diagnostics = self._score_diagnostics(v_k, v_k_safe, hk.view(*gk.shape, -1))

        return gk, diagnostics

    @torch.no_grad()
    def _score_diagnostics(self, v_k, v_k_safe, hk):

        # reshape tensors
        v_k = v_k.view(*hk.shape[:-1], 1)
        v_k_safe = v_k_safe.view(*hk.shape[:-1], 1)

        # compute gamma = log Z - log Z_{-k}
        gamma = self.log_1_m_uniform - torch.log1p(- v_k_safe)

        # vector: gamma hk
        gh = gamma * hk
        # vector: vk hk
        vh = v_k * hk
        # vector: (gamma - vk) hk
        gvh = (gamma - v_k) * hk
        # vector: (gamma - log(1-1/K) - vk) hk
        gmuvh = (gamma - self.log_1_m_uniform - v_k) * hk

        return {'gh': gh, 'vh': vh, 'gvh': gvh, 'gmuvh': gmuvh}

    def forward(self, model: nn.Module,
                x: Tensor,
                backward: bool = False,
                debug: bool = False,
                beta=1.0,
                return_diagnostics: bool = True,
                center: bool = False,
                analysis: bool = False,
                auxiliary_samples:int=0,
                **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        bs = x.size(0)
        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=True, beta=beta, auxiliary_samples=auxiliary_samples)
        L_k, ess, log_wk = [iw_data[k] for k in ('L_k', 'ess', 'log_wk')]

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat([l.view(l.size(0), -1) for l in log_qz], 1).view(bs, self.mc, self.iw, -1)

        if center or analysis:
            hk = self._compute_dlogits(output)
        else:
            hk = None

        prefactor_k, score_diagnostics = self.compute_prefactors(L_k, log_wk, ess, hk=hk, center=center, auxiliary_samples=auxiliary_samples,
                                                                 analysis=analysis, **kwargs)

        log_qz = log_qz[:, :, :prefactor_k.size(2)]

        # compute loss: - \sum_k (log_Z - v_k - c_k) h_k
        reinforce_loss = - torch.sum(prefactor_k[..., None].detach() * log_qz, dim=(2, 3))  # sum over IW and Nz

        # MC averaging
        L_k = L_k.mean(1)
        reinforce_loss = reinforce_loss.mean(1)

        # loss = L_k differentiable w.r.t \theta and reinforce_loss differentiable w.r.t \phi
        loss = - L_k + reinforce_loss

        if analysis:
            output.update(**score_diagnostics)

        # compute dlogits
        if debug:
            output['dqlogits'] = self._compute_dlogits(output)
            output.update(**iw_data)

        # prepare diagnostics
        diagnostics = Diagnostic()

        if return_diagnostics:
            diagnostics.update({
                'loss': {
                    'loss': loss,
                    **self._loss_diagnostics(**iw_data, **output)
                },
                'reinforce': {
                    'loss': reinforce_loss,
                },
            })

            diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
