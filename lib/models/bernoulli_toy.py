from torch import nn

from .base import Template
from ..distributions import *
from ..utils import *
from ..distributions import PseudoBernoulli

import matplotlib.pyplot as plt

class BernoulliToyModel(Template):
    """ Trivial model for the toy example from "Backpropagation Through the Void" paper
    https://arxiv.org/pdf/1711.00123.pdf
    using all estimators """

    def __init__(self, N, K, hdim, **kwargs):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.device_count() else "cpu"
        self.num_latents = N
        self.qlogits = nn.Parameter(torch.zeros(N, requires_grad=True, device=self.device), requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.prior_logits = torch.zeros_like(self.qlogits, device=self.device)
        self.prior_dist = PseudoBernoulli

    def forward(self, x, tau=0, zgrads=False, **kwargs):
        bs = x.shape[0]

        # q_theta(z | x)
        self.qlogits.retain_grad()
        qz = Bernoulli(logits=self.qlogits)
        # prior: p(z)
        pz = Bernoulli(logits=self.prior_logits)

        # sample posterior
        u = torch.rand(bs, self.num_latents).to(self.device)  # u ~ Unif[0,1]
        z = self.qlogits + torch.log(u) - torch.log1p(-u)  # reparametrizing samples, appendix B
        z = z.gt(0.).type_as(z)  # Bernoulli distributed w/ logits "px_logits". Denoted "b" in toy example from Backpropagation Through The Void.

        # p(x|z)
        px = FakeToyDist(b=z)

        diagnostics = self._get_diagnostics(x)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': [self.qlogits], **diagnostics}

    def sample_from_prior(self, N, from_optimal=False):
        # Hack for compatability with run script
        pz = Bernoulli(logits=self.prior_logits)
        z = pz.sample()
        return {'px': None, 'z': z}

    @torch.no_grad()
    def _get_diagnostics(self, x):
        # Calculates loss used in plot for toy problem
        thetas = torch.sigmoid(self.qlogits.detach())
        plot_loss = thetas * (1 - x) ** 2 + (1 - thetas) * x ** 2  # Loss = E_p [ f(x) ], f(x) = (x-t)**2
        plot_loss = plot_loss.mean()
        return {'bernoulli_loss': plot_loss}

    def infer(self, x, tau=0):
        """
        infer the approximate posterior q(z|x) (used for Relax estimator)
        :param x: observation
        :param tau: temperature parameter when using relaxation-based methods (e.g. Gumbel-Softmax)
        :return: posterior distribution, meta data that will be passed in the output
        """

        qlogits = self.get_logits(x)[:, None].repeat(x.shape[0], 1).view(-1, self.num_latents)
        self.prior = torch.zeros_like(qlogits, device=self.device) # For compatability with relax estimator

        diagnostics = self._get_diagnostics(x)

        return qlogits, {'qlogits': [self.qlogits], **diagnostics}

    def get_logits(self, x):
        """ Used for Relax estimator """
        logits = self.qlogits
        logits.retain_grad()
        return logits

    def generate(self, z):
        """ Used for Relax estimator """
        return FakeToyDist(b=z)
