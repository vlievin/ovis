from typing import *

import numpy as np
from torch import nn, Tensor

from .freebits import FreeBits

_EPS = 1e-18


class Estimator(nn.Module):
    def __init__(self, mc: int = 1, iw: int = 1, auxiliary_samples: int = 0, sequential_computation=False,
                 freebits=None, **kwargs):
        """
        A base gradient estimator class.
        :param mc: number of Monte Carlo samples: i.e. \hat{f} = 1/M \sum_m f[m]
        :param iw: number of Importance-weighted samples (see VariationInference class)
        :param auxiliary_samples: number of auxiliary samples used for OVIS MC
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
        self.log_iw = np.log(iw)
        self.log_mc = np.log(mc)
        self.log_mc_iw_m1 = np.log(mc * iw - 1)
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

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs) -> Tuple[Tensor, Dict, Dict]:
        """
        Compute the loss given the `model` and a batch of data `x`. Returns the loss per datapoint, diagnostics and the model's output
        :param model: nn.Module
        :param x: batch of data
        :param backward: compute backward pass
        :param kwargs:
        :return: loss, diagnostics, model's output
        """
        raise NotImplementedError
