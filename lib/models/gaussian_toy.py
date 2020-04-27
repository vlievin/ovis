import torch
from torch import nn

from .base import Template
from ..distributions import NormalFromLoc, PseudoCategorical
from ..utils import batch_reduce


class ToyVAE(Template):
    """
    A simple VAE model parametrized by MLPs
    """

    def __init__(self, xdim, N, K, hdim, **kwargs):
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

    def infer(self, x, tau=0, mc=1, iw=1):
        # retain grads in `b` instead of the qlogits for the gradients analysis
        b = self.b[None, :].expand(x.size(0), self.b.shape[0])
        b.retain_grad()

        qlogits = x @ self.A + b
        qlogits.retain_grad()

        if mc > 1 or iw > 1:
            bs, *dims = qlogits.shape
            qlogits_expanded = qlogits[:, None, None, :].expand(x.size(0), mc, iw, *dims).contiguous()
            qlogits_expanded = qlogits_expanded.view(-1, *dims)

        else:
            qlogits_expanded = qlogits

        qz = self.prior_dist(logits=qlogits_expanded, scale=self.q_scale)
        meta = {'qlogits': [qlogits], 'b': [b]}

        return qz, meta

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs):

        qz, meta = self.infer(x, tau=tau, mc=mc, iw=iw)
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
