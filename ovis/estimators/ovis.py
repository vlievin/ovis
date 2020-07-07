from .vi import *
from .vimco import Vimco


class OvisMonteCarlo(Vimco):
    """Sample based approximation of the Optimal control variate (eq 12)"""

    def __init__(self, iw_aux=1, exclusive=False, **kwargs):
        assert iw_aux > 0  # The `S` samples used to compute the control variate.
        super().__init__(iw_aux=iw_aux, exclusive=exclusive, **kwargs)

    def forward(self,
                model: nn.Module,
                x: Tensor,
                backward: bool = False,
                return_diagnostics: bool = True,
                **kwargs: Any) -> Tuple[Tensor, Diagnostic, Dict]:

        # update the `config` object with the `kwargs`
        config = self.get_runtime_config(**kwargs)
        exclusive, iw, iw_aux = [config.pop(k) for k in ('exclusive', 'iw', 'iw_aux')]

        # define the total number of samples to process
        if exclusive:
            iw_total = iw + iw_aux
        else:
            iw_total = iw
            iw = max(1, iw - iw_aux)

        # forward pass and eval of the log probabilities
        log_probs, output = self.evaluate_model(model, x, iw=iw_total, **config)

        # compute log wk
        log_wk_all = self.compute_log_weights(**log_probs, **config)

        # split log_wk / auxiliary log_wk
        log_wk, log_wk_aux = log_wk_all[:, :, :iw], log_wk_all[:, :, iw:]

        # compute log q(z|x) for the `iw` samples
        log_qz = log_probs['log_qz'][:, :, :iw]

        # compute the score function d_k = \log Z - v_k
        d_k = self.compute_dk(log_wk, **config)

        # compute the control variate c_k
        c_k = self.compute_control_variate(x, log_wk=log_wk, log_wk_aux=log_wk_aux, **config)

        # compute the loss for the inference network (sum over `iw` and `log q(z|x) groups`, avg. over `mc`)
        loss_phi = - ((d_k - c_k).detach() * log_qz.sum(3)).sum(2).mean(1)

        # compute the loss for the generative model and the `iw_data
        if exclusive:
            L_k = self.compute_iw_bound(log_wk=log_wk, **config).mean(1)
        else:
            L_k = self.compute_iw_bound(log_wk=log_wk_all, **config).mean(1)

        loss_theta = - L_k

        # compute diagnostics
        elbo = log_wk.mean(dim=(1, 2))
        ess = self.effective_sample_size(log_wk)
        iw_data = {'elbo': elbo, 'ess': ess, 'kl_q_p': L_k - elbo}

        # compute control variate and MSE
        control_variate_loss = self.compute_control_variate_loss(d_k, c_k)

        # final loss
        loss = loss_theta + loss_phi + self.control_loss_weight * control_variate_loss
        loss = loss.mean()

        # prepare diagnostics
        diagnostics = Diagnostic()
        if return_diagnostics:
            diagnostics = self.base_loss_diagnostics(**iw_data, **log_probs)
            diagnostics.update(self.additional_diagnostics(**output))
            diagnostics.update({'reinforce': {'mse': control_variate_loss}})

        if backward:
            loss.backward()

        return loss, diagnostics, output

    @torch.no_grad()
    def compute_dk(self, log_wk: Tensor, alpha: float = 0, **kwargs):
        # compute the score function d_k = \log Z - v_k
        v_k = log_wk.softmax(2)
        L_k = self.compute_iw_bound(log_wk=log_wk, alpha=alpha, **kwargs)
        return L_k[..., None] - v_k

    @torch.no_grad()
    def compute_control_variate(self,
                                x: Tensor,
                                log_wk: Tensor = None,
                                log_wk_aux: Tensor = None,
                                **kwargs: Dict[str, Tensor]) -> Tensor:
        """
        Compute the control variate using the equation (12):
          * c_k = 1/S \sum_s d_k (z^(s), z_{-k})

        :param x: observation
        :param data: additional data
        :return: control variate
        """

        M, K, S = log_wk.shape[1], log_wk.shape[2], log_wk_aux.shape[2]
        log_wk = log_wk.view(-1, M, 1, 1, K).expand(-1, M, S, K, K)
        log_wk_aux = log_wk_aux.view(-1, M, S, 1, 1).expand(-1, M, S, K, K)

        # for each s, log w_k(z^(s), z_{-k})
        mask = torch.eye(K, device=log_wk.device)
        mask = mask.view(1, 1, 1, K, K)
        log_wk_hat = (1 - mask) * log_wk + mask * log_wk_aux

        # d_k (z^(s), z_{-k})
        d_k_hat = self.compute_dk(log_wk_hat, **kwargs)

        # 1/S \sum_s d_k (z^(s), z_{-k})
        d_k_hat = d_k_hat.diagonal(dim1=3, dim2=4)
        return d_k_hat.mean(2)


class OvisAsymptotic(Vimco):
    """Unified expression of the optimal asymptotic control variate (eq 17)"""

    def __init__(self, gamma: float = 1, **kwargs):
        super().__init__(gamma=gamma, **kwargs)

    @torch.no_grad()
    def compute_control_variate(self,
                                x: Tensor,
                                gamma: float = 1,
                                L_k: Tensor = None,
                                v_k: Tensor = None,
                                **kwargs: Dict[str, Tensor]) -> Tensor:
        """
        Compute the control variate using the equation (17):
          * c_k = log Z_{-k} - gamma v_k + (1-gamma) log (1-1/K)

        The expression log_Z - log Z_{-k} factorizes into (avoids handling the K x K  matrix in memory to compute log Z_{-k})
          * logZ - logZ_{-k} = log \frac{1 - 1/K}{1 - v_k}

        :param x: observation
        :param gamma: parameter for the ESS limit case: gamma==1: ESS >> 1, gamma==0: ESS \approx 1
        :param kwargs: additional data
        :return: control variate
        """
        # avoid overflow: [warning] using a large epsilon value is equivalent to "truncated importance sampling"
        one_minus_v_k = (1 - v_k).clamp(min=torch.finfo(v_k.dtype).eps)

        # log (1 - 1/K)
        log_1_m_uniform = self.cast_log(1 - 1 / v_k.shape[2], v_k)

        # c_k = L_k - (L_k - L_[-k]) - gamma v_k + (1-gamma) log(1 - 1/K)
        logZ_diff = log_1_m_uniform - one_minus_v_k.log()
        return L_k[:, :, None] - logZ_diff - gamma * v_k + (1 - gamma) * log_1_m_uniform


class OvisAsymptoticFromVimco(OvisAsymptotic):
    """
    `OvisAsymptotic` using the `geometric` or `arithmetic` average for `\hat{Z}_{[-k]}`.
    Not tested in the original paper. The `arithmetic` averge corresponds to the `OvisAsymptotic` wihtout using
    the factorization trick.
    """

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, alpha: float = 0, gamma: float = 1, arithmetic: bool = False,
                                **data: Dict[str, Tensor]) -> Tensor:
        v_k = data['v_k']
        log_Z_no_k = Vimco.compute_control_variate(self, x, arithmetic=arithmetic, **data)
        return log_Z_no_k - gamma * v_k + (1 - gamma) * self.log_1_m_uniform
