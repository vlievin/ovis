from typing import *

from copy import copy
import numpy as np
import torch
from booster import Diagnostic
from torch import nn, Tensor

EPS = 1e-18

class Estimator(nn.Module):
    def __init__(self,
                 mc: int = 1,
                 iw: int = 1,
                 pathwise: bool = False,
                 sequential_computation: bool = False,
                 freebits: Optional[float] = None,
                 **kwargs: Any):
        """
        A base gradient estimator class.
        :param mc: number of Monte Carlo samples: i.e. `\hat{L_1} = 1/M \sum_m \log p(x, z_m) / q(z_m | x)`
        :param iw: number of Importance-weighted samples: i.e. `L_K = \log 1/K p(x, z_k) / q(z_k | x) `
        :param sequential_computation: compute each iw sample sequentially (save memory during evaluation)
        :param pathwise: use the reparameterization (allows gradients through `z ~ q(z|x)`)
        :param freebits: [https://arxiv.org/abs/1606.04934]
        :param kwargs: additional keyword arguments
        """
        super().__init__()
        assert mc >= 1
        assert iw >= 1
        self.config = {
            'mc': mc,
            'iw': iw,
            'freebits': freebits,
            'pathwise': pathwise,
            'sequential_computation': sequential_computation,
            **kwargs
        }

    def get_runtime_config(self, **kwargs):
        """update `self.config` with kwargs"""
        if len(kwargs):
            config = copy(self.config)
            config.update(**kwargs)
            return config
        else:
            return copy(self.config)

    @staticmethod
    def _expand_sample(x, mc, iw):
        bs, *dims = x.size()
        x = x[:, None, None].expand(bs, mc, iw, *dims).contiguous()
        # flatten everything into the batch dimension
        return x.view(-1, *dims)

    @staticmethod
    def _reduce_sample(x, mc, iw):
        _, *dims = x.size()
        x = x.view(-1, mc, iw, *dims)
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
