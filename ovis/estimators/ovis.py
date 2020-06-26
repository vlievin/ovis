from typing import Dict

import torch
from torch import Tensor

from ovis import Reinforce


class OvisMonteCarlo(Reinforce):
    """Sample based approximation of the Optimal control variate (eq 12)"""

    def __init__(self, *args, auxiliary_samples=1, **kwargs):
        super().__init__(*args, auxiliary_samples=auxiliary_samples, **kwargs)
        self.baseline = None
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
