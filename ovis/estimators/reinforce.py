from .vi import *


class Reinforce(VariationalInference):
    """
   The "Reinforce" or "score-function" gradient estimator. The gradients of the generative model are given by
      * d L_K/ d\theta = \E_q(z_1, ... z_K | x) [ d/d\theta \log Z ]

    The gradients of the inference network are given by:
      * d L_K/ d\phi = \E_q(z_1, ... z_K | x) [ d_k h_k ]

    Where
      * d_k = \log Z - v_k
      * v_k = w_k / \sum_l w_l
      * h_k = d/d\phi \log q(z_k | x)

    Using a control variate c_k, the gradient estimator g of the parameters of the inference network is:
      * g = \sum_k (d_k - c_k) h_k

    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.detach_qlogits = True  # prevents propagating the gradients through log q(z|x) when differentiating L_k
        self.control_loss_weight = 1. if baseline is not None else 0.
        assert self.sequential_computation == False

    def compute_control_variate(self, x: Tensor, **data: Dict[str, Any]) -> Union[float, Tensor]:
        """
        Compute the baseline (a.k.a control variate). Use the Neural Baseline model if available else the baseline is zero.
        :param x: observation
        :param data: additional data
        :return: control variate of shape [bs, mc, iw]
        """

        if self.baseline is None:
            return 0.

        return self.baseline(x).view((x.size(0), 1, 1))

    def compute_control_variate_loss(self, d_k, c_k):
        """MSE between the score function and the control variate"""
        return (c_k - d_k.detach()).pow(2).mean(dim=(1, 2))

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, debug: bool = False,
                return_diagnostics: bool = True, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:

        bs = x.size(0)
        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        output.update(**self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=self.detach_qlogits, **kwargs))

        # unpack data
        L_k, kl, log_wk = [output[k] for k in ('L_k', 'kl', 'log_wk')]

        # reshape `log_qz` and exclude the auxiliary samples
        log_qz = torch.cat([l.view(l.size(0), -1) for l in log_qz], 1)
        log_qz = log_qz.view(bs, self.mc, self.iw + self.auxiliary_samples, -1)[:, :, :self.iw]

        # compute the score function d_k = \log Z - v_k
        v_k = log_wk.softmax(2)
        d_k = L_k[:, :, None] - v_k

        # compute the control variate c_k
        c_k = self.compute_control_variate(x, d_k=d_k, v_k=v_k, **output, **kwargs)

        # compute the loss for the inference network
        loss_phi = - ((d_k - c_k).unsqueeze(-1).detach() * log_qz).sum(dim=(2, 3)).mean(1)

        # compute the loss for the generative model
        loss_theta = - L_k.mean(1)

        # compute control variate and MSE
        control_variate_loss = self.compute_control_variate_loss(d_k, c_k)

        # final loss
        loss = loss_theta + loss_phi + self.control_loss_weight * control_variate_loss

        if torch.isnan(loss).sum() > 0:
            exit()

        # prepare diagnostics
        diagnostics = Diagnostic()

        if return_diagnostics:
            diagnostics.update({
                'loss': {
                    'loss': loss,
                    **self.base_loss_diagnostics(**output)
                },
                'reinforce': {
                    'mse': control_variate_loss
                }
            })

            diagnostics.update(self.additional_diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output


class OvisMonteCarlo(Reinforce):
    """Sample based approximation of the Optimal control variate (eq 12)"""

    def __init__(self, *args, auxiliary_samples=1, **kwargs):
        super().__init__(*args, auxiliary_samples=auxiliary_samples, **kwargs)
        self.baseline = None
        self.log_iw_m1 = np.log(self.iw - 1)
        self.control_loss_weight = 0

        # The `S` samples used to compute the control variate.
        # The auxiliary samples are not use to learn the generative model, although it is straightforward to do so.
        assert auxiliary_samples > 0

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, alpha: float = 0, **data: Dict[str, Tensor]) -> Tensor:
        """
        Compute the control variate using the equation (12):
          * c_k = 1/S \sum_s d_k (z^(s), z_{-k})

        :param x: observation
        :param alpha: parameter for the IWR bound
        :param data: additional data
        :return: control variate
        """
        L_k, log_wk, log_wk_aux = [data[k] for k in ['L_k', 'log_wk', 'log_wk_aux']]
        K, S = self.iw, self.auxiliary_samples
        log_wk = log_wk.view(-1, self.mc, 1, 1, K).expand(-1, self.mc, S, K, K)
        log_wk_aux = log_wk_aux.view(-1, self.mc, S, 1, 1).expand(-1, self.mc, S, K, K)

        # for each s, log w_k(z^(s), z_{-k})
        mask = torch.eye(K, device=log_wk.device)
        mask = mask.view(1, 1, 1, K, K)
        log_wk_hat = (1 - mask) * log_wk + mask * log_wk_aux

        # compute L_K ((z^(s), z_{-k})) and v_k(z^(s), z_{-k})
        v_k_hat = log_wk_hat.softmax(-1)
        L_k_hat = torch.logsumexp(log_wk_hat, dim=-1, keepdim=True) - self.log_iw
        L_k_hat /= (1. - alpha)  # IWR bound

        # d_k (z^(s), z_{-k})
        d_k_hat = L_k_hat - v_k_hat
        d_k_hat = d_k_hat.diagonal(dim1=3, dim2=4)

        # 1/S \sum_s d_k (z^(s), z_{-k})
        d_k_hat = d_k_hat.mean(2)

        return d_k_hat


class OvisAsymptotic(Reinforce):
    """Unified expression of the optimal asymptotic control variate (eq 17)"""

    def __init__(self, *args, auxiliary_samples=0, **kwargs):
        super().__init__(*args, auxiliary_samples=auxiliary_samples, **kwargs)
        self.baseline = None
        self.log_iw_m1 = np.log(self.iw - 1)
        self.log_1_m_uniform = np.log(1. - 1. / self.iw)
        self.control_loss_weight = 0

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, alpha: float = 0, gamma: float = 1,
                                **data: Dict[str, Tensor]) -> Tensor:
        """
        Compute the control variate using the equation (17):
          * c_k = log Z_{-k} - gamma v_k + (1-gamma) log (1-1/K)

        The expression log_Z - log Z_{-k} factorizes into (avoids handling the K x K  matrix in memory to compute log Z_{-k})
          * logZ - logZ_{-k} = log \frac{1 - 1/K}{1 - v_k}

        :param x: observation
        :param alpha: parameter for the IWR bound
        :param gamma: parameter for the ESS limit case: gamma==1: ESS >> 1, gamma==0: ESS \approx 1
        :param data: additional data
        :return: control variate
        """
        L_k, v_k = [data[k] for k in ['L_k', 'v_k']]
        # avoid overflow
        v_k_safe = torch.min((1 - 1e-7) * torch.ones_like(v_k), v_k)

        # c_k = L_k
        logZ_diff = self.log_1_m_uniform - torch.log1p(- v_k_safe)
        return L_k[:, :, None] - logZ_diff - gamma * v_k + (1 - gamma) * self.log_1_m_uniform


class Vimco(Reinforce):
    """
    VIMCO: https://arxiv.org/abs/1602.06725
    Inspired by https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = None
        self.log_iw_m1 = np.log(self.iw - 1)

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, arithmetic=False, use_outer_samples=False, use_double: bool = False,
                                **data: Dict[str, Tensor]) -> Tensor:
        """
        Compute the Vimco control variate `c_k = c_k(z_{-k}) = log 1/k \sum_{l \neq k} w_l + \hat{w}_{-k}`
          * arithmetic: \hat{w}_{-k} = 1/K-1 \sum_{l \neq k} w_l
          * geometric:  \hat{w}_{-k} = exp( 1/K-1 \sum_{l \neq k} log w_l )

        When `use_outer_samples == True` with M>1 outer samples, the average exploits the `MK - 1` independent z samples

        :param x: observation
        :param arithmetic: average type
        :param use_outer_samples: also use the M outer MC samples in the computation of `\hat{w}_{-k}`
        :param use_double: use double precision
        :param data: additional data
        :return: control variate `c_k`
        """

        log_wk = data['log_wk']
        log_wk = log_wk.view(-1, self.mc, self.iw)
        _dtype = log_wk.dtype
        log_mc_iw_m1 = np.log(self.iw - 1)

        if self.iw == 1:
            return torch.zeros_like(log_wk[:, :, 0])

        if use_double:
            log_wk = log_wk.double()

        if arithmetic:  # arithmetic: \hat{w}_{-k} = 1/K-1 \sum_{l \neq k} w_l

            if use_outer_samples:

                mask = 1 - torch.eye(self.iw * self.mc, dtype=log_wk.dtype, device=log_wk.device)[None, :, :]
                _log_wk = log_wk[:, None, None, :, :].expand(-1, self.mc, self.iw, self.mc, self.iw)
                _log_wk = _log_wk.view(log_wk.size(0), self.mc * self.iw, self.mc * self.iw)

                # make sure to replace the masked samples with mins to avoid overflow
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

                # make sure to replace the masked samples with mins to avoid overflow
                _min, _ = _log_wk.min(dim=3, keepdim=True)
                _log_wk = (1 - mask) * _min + mask * _log_wk

                # get the max for the LSE trick
                max, idx = _log_wk.max(dim=3, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_wk - max), dim=3)
                log_wk_hat = max.squeeze(3) + torch.log(sum_exp) - log_mc_iw_m1

        else:  # geometric: \hat{w}_{-k} = exp( 1/K-1 \sum_{l \neq k} log w_l )
            if use_outer_samples:
                log_wk_hat = (torch.sum(log_wk, dim=(1, 2), keepdim=True) - log_wk) / (self.mc * self.iw - 1)
            else:
                log_wk_hat = (torch.sum(log_wk, dim=2, keepdim=True) - log_wk) / (self.iw - 1)

        #  c_k = log 1/k \sum_{l \neq k} w_l + \hat{w}_{-k}
        log_wk_samples = log_wk.unsqueeze(-1) + torch.diag_embed(log_wk_hat - log_wk)
        return torch.logsumexp(log_wk_samples, dim=2) - self.log_iw
