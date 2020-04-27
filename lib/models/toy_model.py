from copy import copy

from typing import Dict

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli

from .models import Template
from ..distributions import *
from ..layers import *
from ..utils import *


class ToyModel(Template):
    """ Trivial model for the toy example from "Backpropagation Through the Void" paper
    https://arxiv.org/pdf/1711.00123.pdf
    using all estimators """
    def __init__(self, num_latents, batch_size, logits = nn.Parameter(torch.zeros(1))): # , mc, iw, logits = nn.Parameter(torch.zeros(1))):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.device_count() else "cpu"
        self.qlogits = nn.Parameter(torch.zeros(num_latents, requires_grad=True, device=self.device), requires_grad=True)
        self.num_latents = num_latents
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.device_count() else "cpu"

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs): # (self, mc=1, iw=1, **kwargs): # (self, *args, **kwargs) -> Dict[str, Tensor]: # (self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        # q_theta(z | x)
        self.qlogits.retain_grad()
        qz = Bernoulli(logits=self.qlogits)

        # prior: p(z)
        pz = Bernoulli(logits=torch.zeros_like(self.qlogits, device=self.device)) # Bernoulli(logits=self.logits)

        # sample posterior
        u = torch.rand(self.batch_size * iw * mc, self.num_latents).to(self.device) # u ~ Unif[0,1]
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
        diagnostics = self._get_diagnostics(z, qz, pz)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': [self.qlogits], **diagnostics}

    def _get_diagnostics(self, z, qz, pz):
        # Hack for compatability with estimators
        Hp = torch.tensor([torch.zeros_like(pz.entropy())])
        Hp = Hp[:,None]
        if isinstance(pz, PseudoCategorical):
            usage = torch.zeros_like(Hp)
        else:
            usage = torch.zeros_like(Hp)

        return {'Hp': [Hp], 'usage': [usage]}

