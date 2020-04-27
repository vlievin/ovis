from copy import copy
from typing import Dict

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli, Normal

from .base import Template
from ..distributions import *
from ..utils import *


class ToyModel(Template):
    """ Trivial model for the toy example from "Backpropagation Through the Void" paper
    https://arxiv.org/pdf/1711.00123.pdf
    using all estimators """
    def __init__(self, xdim, N, K, hdim, **kwargs): # (self, num_latents, batch_size):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.device_count() else "cpu"
        self.num_latents = N
        self.qlogits = nn.Parameter(torch.zeros(N, requires_grad=True, device=self.device), requires_grad=True)
        self.prior_logits = torch.zeros_like(self.qlogits, device=self.device)

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs): # (self, mc=1, iw=1, **kwargs): # (self, *args, **kwargs) -> Dict[str, Tensor]: # (self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        bs = x.shape[0]

        # q_theta(z | x)
        self.qlogits.retain_grad()
        qz = Bernoulli(logits=self.qlogits)

        # prior: p(z)
        pz = Bernoulli(logits=self.prior_logits)

        # sample posterior
        u = torch.rand(bs * iw * mc, self.num_latents).to(self.device) # u ~ Unif[0,1]
        # v = torch.rand(self.batch_size * iw * mc, self.num_latents).to(self.device) # v ~ Unif[0,1]
        z = self.qlogits + torch.log(u) - torch.log1p(-u) # reparametrizing samples, appendix B
        z = z.gt(0.).type_as(z) # Bernoulli distributed w/ logits "px_logits". Denoted "b" in toy example from Backpropagation Through The Void.

        # p(x|z)
        if self.training:
            px = FakeToyDist(b=z, training=False)
        else:
            px = FakeToyDist(b=z, training=False)

        # values from the stochastic layers (z, pz, qz) are returned
        # as a list where each index correspond to one stochastic layer
        diagnostics = self._get_diagnostics(x, qz, pz)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': [self.qlogits], **diagnostics}

    def sample_from_prior(self, N, from_optimal=False):
        # Hack for compatability with run script
        pz = Bernoulli(logits=self.prior_logits)
        z = pz.sample()
        px = Normal(0,1)
        return {'px': None, 'z': z}

    @torch.no_grad()
    def _get_diagnostics(self, x, qz, pz):
        # Calculates loss used in plot for toy problem
        thetas = torch.sigmoid(self.qlogits.detach())# .cpu().numpy()
        plot_loss = thetas * (1 - x) ** 2 + (1 - thetas) * x **2 # Loss = E_p [ f(x) ], f(x) = (x-t)**2
        plot_loss = plot_loss.mean()
        return {'bernoulli_mse': plot_loss}


