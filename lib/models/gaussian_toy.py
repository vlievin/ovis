import torch
from torch import nn

from .base import Template
from ..distributions import NormalFromLoc, PseudoCategorical
from ..utils import batch_reduce


class GaussianToyVAE(Template):
    """
    A simple Gaussian VAE model as defined in
    `Tighter Variational Bounds are Not Necessarily Better` [https://arxiv.org/abs/1802.04537]

    * z ~ \mathcal{N} (z, \mu, I)
    * x|z ~ \mathcal{N} (x ; z, I)
    """

    def __init__(self, xdim: int = 20, **kwargs):
        super().__init__()
        D = xdim[0]
        self.xdim = xdim
        self.prior_dist = NormalFromLoc
        self.likelihood = NormalFromLoc

        self.A = nn.Parameter(0.01 * torch.randn(D, D))
        self.b = nn.Parameter(0.01 * torch.randn(D))
        self.register_buffer("q_scale", 2 / 3 * torch.ones((1, D,)))
        self.mu = nn.Parameter(torch.zeros((1, D,)))

    def generate(self, z):
        return self.likelihood(logits=z)

    def infer(self, x, tau=0):
        # retain grads in `b` instead of the qlogits for the gradients analysis
        b = self.b[None, :].expand(x.size(0), self.b.shape[0])
        b.retain_grad()

        qlogits = x @ self.A + b
        qlogits.retain_grad()

        qz = self.prior_dist(logits=qlogits, scale=self.q_scale)
        meta = {'qlogits': [qlogits], 'b': [b]}

        return qz, meta

    def forward(self, x, tau=0, zgrads=False, **kwargs):

        qz, meta = self.infer(x, tau=tau)
        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = self.prior_dist(logits=self.mu)

        px = self.generate(z)

        diagnostics = self._get_diagnostics(z, qz, pz)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], **meta, **diagnostics}

    def sample_from_prior(self, N):
        prior = self.mu.expand(N, *self.xdim)
        z = self.prior_dist(logits=prior).sample()
        px = self.generate(z)
        return {'px': px, 'z': z}

    def _get_diagnostics(self, z, qz, pz):
        Hp = batch_reduce(pz.entropy())
        if isinstance(pz, PseudoCategorical):
            usage = (z.sum(dim=0, keepdim=True) > 0).float()
            usage = usage.mean(dim=(1, 2,))
        else:
            usage = torch.zeros_like(Hp)

        return {'Hp': [Hp], 'usage': [usage]}
