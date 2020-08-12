from typing import *

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal, Bernoulli
from torch.distributions import constraints
from torch.nn.functional import gumbel_softmax, log_softmax


class BaseDistribution(Distribution):
    """A base class wrapper of torch.Distributions that takes a single tensor `logits` as parameter"""

    arg_constraints = {'logits': constraints.real}

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1, **kwargs):
        self.logits = logits
        self.tau = tau
        self.dim = dim

    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError


class PseudoCategorical(BaseDistribution):
    """
    Categorical distribution when `tau=0`, else Gumbel-Softmax relaxation. This is not the Gumbel-Softmax distribution
    because the `log_prob` is evaluated differently.
    """

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1, **kwargs):
        logits = logits - logits.logsumexp(dim=dim, keepdim=True)
        super().__init__(logits, tau=tau, dim=dim)

    def rsample(self):

        if self.tau == 0:
            hard = True
            tau = 0.5
        else:
            hard = False
            tau = self.tau

        return gumbel_softmax(self.logits, tau=tau, hard=hard, dim=self.dim)

    def sample(self):
        with torch.no_grad():
            return self.rsample()

    def entropy(self):
        return - (self.logits * self.logits.exp()).sum(dim=self.dim)

    def log_prob(self, value):
        log_pdf = log_softmax(self.logits, self.dim)
        return (value * log_pdf).sum(self.dim)


class PseudoBernoulli(Bernoulli):

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1, **kwargs):
        super().__init__(logits=logits)
        assert tau == 0, 'Not implemented for tau > 0'
        self.tau = tau
        self.dim = dim

    def sample(self, **kwargs):
        return super().sample(**kwargs)

    def rsample(self, **kwargs):
        raise NotImplementedError


class NormalFromLogits(BaseDistribution):
    """
    Defines a Normal distribution from a `logits` tensors where `mu, log_std = logits.chunk(2)`
    """

    def __init__(self, logits: Tensor, dim: int = -1, **kwargs: Any):
        super().__init__(logits, dim=dim, **kwargs)

    @property
    def _params(self):
        # logits are of dimension (*, N, K,) with K = 2
        mu, log_std = self.logits.chunk(2, dim=self.dim)
        scale = log_std.exp()
        return mu.squeeze(self.dim), scale.squeeze(self.dim)

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
    """
    Defines a normal distribution from the location parameter only, the scale is assume to be 1 if not provided.
    """

    def __init__(self, logits: Tensor, scale=None, dim: int = -1, **kwargs: Any):
        """hacking the Normal class so we can easily compute d p.log_prob(z) / d logits"""
        super(Distribution, self).__init__()
        self.logits = logits
        self.scale = torch.ones_like(logits) if scale is None else scale
        self.dim = dim

    @property
    def _torch_normal(self):
        return Normal(loc=self.logits, scale=self.scale)
