from typing import Dict

import numpy
import torch
from torch import Tensor

from ovis import Reinforce


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