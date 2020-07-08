from typing import *

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli, Distribution

from ovis.models.distributions import PseudoCategorical
from ovis.utils.utils import prod, flatten


def H(z, dim=-1):
    index = z.max(dim, keepdim=True)[1]
    return torch.zeros_like(z).scatter_(dim, index, 1.0)


class TemplateModel(nn.Module):
    """A template of VAE model, follow these guidelines to make your model compatible
    with the gradient estimators and the training loop"""

    def forward(self, x: Tensor,
                reparam: bool = True,
                **kwargs) -> Dict[str, Union[Tensor, List[Tensor], List[Distribution], Distribution]]:
        """
        Perform a forward pass and return the latent samples `z` and the distribution `p(x|z)`, `p(z)`, `q(z|x)`.

        `p(z)`, `q(z|x)` and `z` are returned as lists where each element represent a "group" of latent variables. This
        is useful to group latent variables per layer in hierarchical models. So statistics can be access individually
        for each group. In this repo, the `Freebits` are applied independently for each "group".

        **warning**: The gradient estimators depends on the argument `reparam` that indicates if the reparameterization
        has to be used.

        :param x: observation
        :param reparam: use reparameterization
        :param kwargs: other arguments
        :return: {'px': p(x|z), 'z': [z_i], 'pz': [p(z_i|*)], 'qz': [q(z_i|x,*)]}
        """

        raise NotImplementedError

        bs, *dims = (16, 1, 8, 8)
        N = K = 4

        # input data
        x = torch.rand(bs, *dims)

        # posterior: q(z|x)
        encoder = nn.Linear(prod(dims), N * K)
        qlogits = encoder(flatten(x)).view(ns, N, K)
        qz = PseudoCategorical(logits=qlogits)

        # prior: p(z)
        plogits = torch.zeros(bs, N, K)
        pz = PseudoCategorical(logits=plogits, **kwargs)

        # sample posterior
        z = qz.rsample() if reparam else qz.sample()

        # p(x|z)
        decoder = nn.Linear(N * K, prod(dims))
        px_logits = decoder(flatten(z)).view(bs, *dims)
        px = Bernoulli(logits=px_logits)

        # Values from the stochastic layers (z, pz, qz) are returned
        # as a list where each index correspond to one group of latent variables
        return {'px': px,  # Distribution: p(x|z)
                'z': [z],  # List[Tensor]: z ~ q(z|x)
                'qz': [qz],  # List[Distribution]: q(z|x)
                'pz': [pz]  # List[Distribution]: p(z)
                }

    def sample_from_prior(self, bs: int, **kwargs):
        """sample from the prior and return the distribution `p(x|z)`"""

        raise NotImplementedError

        bs, *dims = (16, 1, 8, 8)
        N = K = 4

        # prior: p(z)
        plogits = torch.zeros(bs, N, K)
        pz = PseudoCategorical(logits=plogits, **kwargs)

        # sample prior
        z = pz.rsample()

        # p(x|z)
        decoder = nn.Linear(N * K, prod(dims))
        px_logits = decoder(flatten(z)).view(bs, *dims)
        px = Bernoulli(logits=px_logits)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz]}
