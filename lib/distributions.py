from typing import *

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.nn.functional import gumbel_softmax, log_softmax


class PseudoCategorical(Distribution):

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1):
        self.logits = logits - logits.logsumexp(dim=dim, keepdim=True)
        self.dim = dim
        self.tau = tau

    def rsample(self):

        if self.tau == 0:
            hard = True
            tau = 0.5
        else:
            hard = False
            tau = self.tau

        return gumbel_softmax(self.logits, tau=tau, hard=hard, dim=self.dim)

    def sample(self):
        return self.rsample().detach()

    def entropy(self):
        return - (self.logits * self.logits.exp()).sum(dim=self.dim)

    def log_prob(self, value):
        log_pdf = log_softmax(self.logits, self.dim)
        return (value * log_pdf).sum(self.dim)


class NormalFromLogits(Distribution):
    def __init__(self, logits: Tensor, dim: int = -1, **kwargs: Any):
        """hacking the Normal class so we can easily compute d p.log_prob(z) / d logits"""
        super().__init__()
        self.logits = logits
        self.dim = dim

    @property
    def _params(self):
        # logits are of dimension (*, N, K,) with K = 2
        mu, log_std = self.logits.chunk(2, dim=self.dim)
        scale = log_std.mul(0.5).exp()
        return mu, scale

    @property
    def _torch_normal(self):
        return Normal(*self._params)

    def sample(self):
        return self._torch_normal.sample()

    def rsample(self):
        return self._torch_normal.rsample()

    def entropy(self):
        return self._torch_normal.entropy()

    def log_prob(self, x):
        return self._torch_normal.log_prob(x)


class NormalFromLoc(NormalFromLogits):
    def __init__(self, logits: Tensor, scale=None, dim: int = -1, **kwargs: Any):
        """hacking the Normal class so we can easily compute d p.log_prob(z) / d logits"""
        super(Distribution, self).__init__()
        self.logits = logits
        self.scale = torch.ones_like(logits) if scale is None else scale
        self.dim = dim

    @property
    def _torch_normal(self):
        return Normal(loc=self.logits, scale=self.scale)

    
class FakeToyDist(Distribution):

    """ Fake probability distribution used for Toy example (run_bernoulli_toy.py)
        Usable with various estimators such that score function is MSE. """

    def __init__(self, b: Tensor, training: bool):
        super().__init__()
        self.b = b
        self.training = training

    def log_prob(self, target):
        if self.training:
            target = target[0] # Workaround for _expand_sample in base Estimator class
        MSE = (self.b-target)**2 # log score ( log f_b )
        return MSE

    def rsample(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
