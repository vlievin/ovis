import torch
from torch import nn

from .base import Template
from .vae import BaseVAE
from ..distributions import PseudoCategorical, NormalFromLoc


class GaussianMixture(BaseVAE):
    """
    A simple VAE model parametrized by MLPs
    """

    def __init__(self, xdim, C, K, hdim=16, **kwargs):
        super(Template, self).__init__()
        act = nn.Tanh
        xdim = 1
        self.C = C
        self.register_buffer('log_theta_opt', torch.log(5 + torch.arange(0, C, dtype=torch.float)).view(1, 1, C))
        self.log_theta = nn.Parameter(torch.zeros(1, 1, C))
        self.prior_dist = PseudoCategorical
        self.likelihood = NormalFromLoc
        self.register_buffer('p_mu', 10. * torch.arange(0, C))
        self.register_buffer('p_scale', torch.tensor(5.))

        self.phi = nn.Sequential(
            nn.Linear(xdim, hdim),
            act(),
            nn.Linear(hdim, C)
        )

    def true_posterior(self, x):
        M, C = x.size(0), self.C
        x = x.view(M, 1, 1).expand(M, C, 1)
        z = torch.eye(C, device=x.device)[None, :, None, :]
        p_x_z = self.generate(z)
        log_p_x_z = p_x_z.log_prob(x)
        log_posterior = self.log_theta_opt.sum(1) + log_p_x_z.sum(-1)
        return self.prior_dist(log_posterior.log_softmax(dim=-1))

    def generate(self, z):
        z = z.argmax(dim=-1)
        mu = self.p_mu[z]
        return self.likelihood(logits=mu, scale=self.p_scale)

    def get_logits(self, x):
        logits = self.phi(x.view(-1, 1)).view(-1, 1, self.C)

        logits.retain_grad()

        return logits

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs):

        qz, meta = self.infer(x, tau=tau, mc=mc, iw=iw)

        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = self.prior_dist(logits=self.log_theta)

        px = self.generate(z)

        diagnostics = self._get_diagnostics(x, self.prior_dist(logits=meta['qlogits'][0]))

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], **meta, **diagnostics}

    def sample_from_prior(self, N, from_optimal=False):
        if from_optimal:
            prior = self.log_theta_opt.expand(N, 1, self.C)
        else:
            prior = self.log_theta.expand(N, 1, self.C)
        z = self.prior_dist(logits=prior).sample()
        px = self.generate(z)
        return {'px': px, 'z': z}

    @torch.no_grad()
    def _get_diagnostics(self, x, qz):

        # compute prior mse
        prior_mse = (self.log_theta.softmax(-1) - self.log_theta_opt.softmax(-1)).norm(p=2, dim=-1).mean()

        # compute the posterior MSE
        true_posterior = self.true_posterior(x).logits.softmax(-1)
        posterior = qz.logits.sum(1).softmax(-1)

        posterior_mse = (posterior - true_posterior).norm(p=2, dim=-1).mean()

        return {'prior_mse': prior_mse, 'posterior_mse': posterior_mse}
