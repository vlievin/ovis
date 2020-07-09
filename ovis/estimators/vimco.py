from typing import Dict

import torch
from torch import Tensor

from .reinforce import Reinforce
from ..utils.utils import cast_tensor


class Vimco(Reinforce):
    """
    Variational inference for Monte Carlo objectives (VIMCO) [https://arxiv.org/abs/1602.06725]
    Inspired by [https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py]
    """

    def __init__(self, **kwargs):
        assert kwargs.pop('baseline', None) is None, f"Neural Baselines are not handled for `{type(self).__name__}`"
        super().__init__(baseline=None, **kwargs)

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, log_wk: Tensor, mc: int = 1, iw: int = 1, arithmetic=False,
                                use_double: bool = False, **kwargs: Dict[str, Tensor]) -> Tensor:
        """
        Compute the Vimco control variate `c_k = c_k(z_{-k}) = log 1/k \sum_{l \neq k} w_l + \hat{w}_{-k}`
          * arithmetic: \hat{w}_{-k} = 1/K-1 \sum_{l \neq k} w_l
          * geometric:  \hat{w}_{-k} = exp( 1/K-1 \sum_{l \neq k} log w_l )

        :param x: observation
        :param mc: number of outer Monte-Carlo samples
        :param iw: number of Importance Weighted samples
        :param arithmetic: average type
        :param use_double: use double precision
        :param kwargs: additional data
        :return: control variate `c_k`
        """

        log_wk = log_wk.view(-1, mc, iw)
        _dtype = log_wk.dtype

        if iw == 1:
            return torch.zeros_like(log_wk[:, :, 0])

        if use_double:
            log_wk = log_wk.double()

        if arithmetic:  # arithmetic: \hat{w}_{-k} = 1/K-1 \sum_{l \neq k} w_l
            mask = 1 - torch.eye(iw, dtype=log_wk.dtype, device=log_wk.device)[None, None, :, :]
            _log_wk = log_wk[:, :, None, :].expand(-1, mc, iw, iw)

            # make sure to replace the masked samples with mins to avoid overflow
            _min, _ = _log_wk.min(dim=3, keepdim=True)
            _log_wk = (1 - mask) * _min + mask * _log_wk

            # get the max for the LSE trick
            max, idx = _log_wk.max(dim=3, keepdim=True)

            sum_exp = torch.sum(mask * torch.exp(_log_wk - max), dim=3)
            log_wk_hat = max.squeeze(3) + torch.log(sum_exp) - cast_tensor(iw - 1, log_wk).log()

        else:  # geometric: \hat{w}_{-k} = exp( 1/K-1 \sum_{l \neq k} log w_l )
            log_wk_hat = (torch.sum(log_wk, dim=2, keepdim=True) - log_wk) / (iw - 1)

        #  c_k = log 1/k \sum_{l \neq k} w_l + \hat{w}_{-k}
        log_wk_samples = log_wk.unsqueeze(-1) + torch.diag_embed(log_wk_hat - log_wk)
        c_k = torch.logsumexp(log_wk_samples, dim=2) - cast_tensor(iw, log_wk).log()

        if use_double:
            c_k = c_k.to(_dtype)

        return c_k


class VimcoArithmetic(Vimco):
    """
    Variational inference for Monte Carlo objectives (VIMCO) [https://arxiv.org/abs/1602.06725]
    Inspired by [https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py]

    Arithmetic average:

        * arithmetic: \hat{w}_{-k} = 1/K-1 \sum_{l \neq k} w_l
    """

    def __init__(self, **kwargs):
        assert kwargs.pop('baseline', None) is None, f"Neural Baselines are not handled for `{type(self).__name__}`"
        super().__init__(baseline=None, arithmetic=True, **kwargs)


class VimcoGeometric(Vimco):
    """
    Variational inference for Monte Carlo objectives (VIMCO) [https://arxiv.org/abs/1602.06725]
    Inspired by [https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py]

    Geometric average:

        * geometric:  \hat{w}_{-k} = exp( 1/K-1 \sum_{l \neq k} log w_l )
    """

    def __init__(self, **kwargs):
        assert kwargs.pop('baseline', None) is None, f"Neural Baselines are not handled for `{type(self).__name__}`"
        super().__init__(baseline=None, arithmetic=False, **kwargs)
