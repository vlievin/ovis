from typing import *

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.nn.functional import gumbel_softmax, log_softmax


class BaseDistribution(Distribution):
    """A base class for the distribution used in the project.
    The rest of the framework relies on the """

    def __init__(self, logits: Tensor):
        self.logits = logits

    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError


class PseudoCategorical(BaseDistribution):

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1):
        logits = logits - logits.logsumexp(dim=dim, keepdim=True)
        super().__init__(logits)
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


class PseudoBernoulli(BaseDistribution):

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1):
        super().__init__(logits)
        self.tau = tau

    @property
    def _logits(self):
        params = self.logits.unsqueeze(-1)
        return torch.cat([params, torch.zeros_like(params)], -1).log_softmax(-1)

    def rsample(self):

        if self.tau == 0:
            hard = True
            tau = 0.5
        else:
            hard = False
            tau = self.tau

        z = gumbel_softmax(self._logits, tau=tau, hard=hard, dim=-1)
        return z[..., 0]

    def sample(self):
        return self.rsample().detach()

    def entropy(self):
        logits = self._logits
        return - (logits * logits.exp()).sum(dim=-1)

    def log_prob(self, value):
        log_pdf = log_softmax(self._logits, -1)
        value = torch.cat([value.unsqueeze(-1), 1. - value.unsqueeze(-1)], -1)
        return (value * log_pdf).sum(-1)


class NormalFromLogits(BaseDistribution):
    def __init__(self, logits: Tensor, dim: int = -1, **kwargs: Any):
        """hacking the Normal class so we can easily compute d p.log_prob(z) / d logits"""
        super().__init__(logits)
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
