import numpy as np
from torch import nn, Tensor
from torch.distributions import Categorical, RelaxedOneHotCategorical, Bernoulli, Distribution
from torch.nn.functional import one_hot, gumbel_softmax, log_softmax

from .utils import *


class PseudoCategorical(Distribution):

    def __init__(self, logits: Tensor, tau: float = 0, dim: int = -1):
        self.logits = logits
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

    def log_prob(self, value):
        log_pdf = log_softmax(self.logits, self.dim)
        return (value * log_pdf).sum(self.dim)


def H(z, dim=-1):
    index = z.max(dim, keepdim=True)[1]
    return torch.zeros_like(z).scatter_(dim, index, 1.0)


class VAE(nn.Module):
    def __init__(self, xdim, N, K, hdim, nlayers=0, learn_prior=False, bias=True, normalization='layernorm',
                 likelihood=Bernoulli):
        super().__init__()
        self.xdim = xdim
        zdim = N * K
        self.N = N
        self.K = K
        Norm = {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm, 'none': None, None: None}[normalization]

        # encoder
        layers = []
        h = int(np.prod(xdim))
        for i in range(nlayers):
            layers += [nn.Linear(h, hdim, bias=bias)]
            if Norm is not None:
                layers += [Norm(hdim)]
            layers += [nn.ELU()]
            h = hdim
        layers += [nn.Linear(h, zdim, bias=bias)]

        self.encoder = nn.Sequential(*layers)

        # decoder
        layers = []
        h = zdim
        for i in range(nlayers):
            layers += [nn.Linear(h, hdim, bias=bias)]
            if Norm is not None:
                layers += [Norm(hdim)]
            layers += [nn.ELU()]
            h = hdim
        layers += [nn.Linear(h, int(np.prod(xdim)), bias=bias)]

        self.decoder = nn.Sequential(*layers)

        prior = torch.zeros((1, N, K,))
        if learn_prior:
            self.prior = nn.Parameter(prior)
        else:
            self.register_buffer('prior', prior)

        self.likelihood = likelihood

        # decoder weights
        self.decoder_weights = [v for k, v in self.decoder.named_parameters() if 'weight' in k]

    def infer(self, x):
        x = flatten(x)
        return self.encoder(x).view(-1, self.N, self.K)

    def generate(self, z):
        z = flatten(z)
        px_logits = self.decoder(z).view(-1, *self.xdim)
        return self.likelihood(logits=px_logits)

    def lipschitz(self):
        l = 1
        for w in self.decoder_weights:
            l *= w.abs().max()
        return l

    def forward(self, x, tau=0, zgrads=False):
        qlogits = self.infer(x)

        qz = PseudoCategorical(logits=qlogits, tau=tau)
        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = PseudoCategorical(logits=self.prior)

        px = self.generate(z)

        return {'px': px, 'z': z, 'qz': qz, 'pz': pz, 'qlogits': qlogits}

    def sample_from_prior(self, N):
        prior = self.prior.expand(N, self.N, self.K)
        z = PseudoCategorical(logits=prior).sample()
        px = self.generate(z)
        return {'x_': px, 'z': z}
