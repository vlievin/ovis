from typing import *

import numpy as np
import torch
from booster import Diagnostic
from torch import nn, Tensor

from .freebits import FreeBits

EPS = 1e-18


class Estimator(nn.Module):
    def __init__(self, mc: int = 1,
                 iw: int = 1,
                 auxiliary_samples: int = 0,
                 sequential_computation=False,
                 freebits=None,
                 **kwargs):
        """
        A base gradient estimator class.
        :param mc: number of Monte Carlo samples: i.e. `\hat{L_1} = 1/M \sum_m \log p(x, z_m) / q(z_m | x)`
        :param iw: number of Importance-weighted samples: i.e. `L_K = \log 1/K p(x, z_k) / q(z_k | x) `
        :param auxiliary_samples: auxiliary_samples excluded from the computation of `L_k`.
        :param sequential_computation: compute each iw sample sequentially (save memory during evaluation)
        :param freebits: [https://arxiv.org/abs/1606.04934]
        :param kwargs: additional keyword arguments
        """
        super().__init__()
        assert mc >= 1
        assert iw >= 1
        self.mc = mc
        self.iw = iw
        self.auxiliary_samples = auxiliary_samples
        self.register_buffer('log_iw', torch.tensor(np.log(iw)))
        self.register_buffer('log_mc', torch.tensor(np.log(mc)))
        self.freebits = None if freebits is None else FreeBits(freebits)
        self.detach_qlogits = False
        self.sequential_computation = sequential_computation

    def _expand_sample(self, x):
        bs, *dims = x.size()
        x = x[:, None, None].repeat(1, self.mc, self.iw + self.auxiliary_samples, *(1 for _ in dims))
        # flatten everything into the batch dimension
        return x.view(-1, *dims)

    def _reduce_sample(self, x):
        _, *dims = x.size()
        x = x.view(-1, self.mc, self.iw + self.auxiliary_samples, *dims)
        return x.mean((1, 2,))

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs) -> Tuple[Tensor, Diagnostic, Dict]:
        """
        Compute the loss given the `model` and a batch of data `x`. Returns the loss per datapoint, diagnostics and the model's output
        :param model: nn.Module
        :param x: batch of data
        :param backward: compute backward pass
        :param kwargs:
        :return: loss, diagnostics, model's output
        """
        raise NotImplementedError
