import sys

import torch
from torch import nn

from .base import Template
from ..distributions import NormalFromLoc, Normal


class SafeSeed():
    """
    A small class to execute code with a gicen random seed without breaking the randomness
    """

    def __init__(self, seed):
        # seed to for the enter during the enter
        self.seed = seed

        # seed to set for the exit
        self.exit_seed = int(torch.randint(1, sys.maxsize, (1,)).item())

    def __enter__(self):
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        self.exit_seed = int(torch.randint(1, sys.maxsize, (1,)).item())


class GaussianToyVAE(Template):
    """
    A simple Gaussian VAE model as defined in
    `Tighter Variational Bounds are Not Necessarily Better` [https://arxiv.org/abs/1802.04537]

    * z ~ \mathcal{N} (z, \mu, I)
    * x|z ~ \mathcal{N} (x ; z, I)

    The true model is seeded with a fixed random seed `true_model_seed`.
    All parameters are initialized form U[1.5, 2.5]
    """

    def __init__(self, xdim: int = 20, npoints=1024, true_model_seed=42, **kwargs):
        super().__init__()
        self.true_model_seed = true_model_seed
        self.npoints = npoints
        self.D = xdim[0]
        self.xdim = xdim
        self.prior_dist = NormalFromLoc
        self.likelihood = NormalFromLoc

        # inference model
        self.A = nn.Parameter(1.5 + torch.rand(self.D, self.D))
        self.b = nn.Parameter(1.5 + torch.rand(self.D))
        self.register_buffer("q_scale", 2 / 3 * torch.ones((1, self.D,)))

        # generative model
        self.mu = nn.Parameter(1.5 + torch.rand((1, self.D,)))

        # store the dataset and the optimal parameters
        self.register_buffer("dset", self.sample_dset())
        for k, v in self.get_optimal_parameters(self.dset).items():
            self.register_buffer(f"opt_{k}", v)

    def sample_mu_true(self):
        """get the optimal paramter \mu^{true} given the model true seed"""
        with SafeSeed(self.true_model_seed):
            mu_true = Normal(loc=torch.zeros((self.D,)), scale=torch.ones((self.D,))).sample()

        return mu_true

    def sample_dset(self):
        """sample the data set given \mu^{true}, the number of dset data points and the model random seed"""
        mu = self.sample_mu_true()
        with SafeSeed(self.true_model_seed):
            dset = self.sample_from_prior(self.npoints, mu=mu)['px'].sample()

        return dset

    def get_optimal_parameters(self, dset):
        """compute the otpimal parameters given the dataset"""
        mu = dset.mean(dim=0, keepdim=True).data  # \mu^* = 1/N \sum_k x_k
        A = 0.5 * torch.eye(dset.shape[1], device=dset.device).view_as(self.A)  # A = I / 2
        b = 0.5 * mu.view_as(self.b.data)  # b = 0.5 mu^*

        return {'mu': mu, 'A': A, 'b': b}

    def set_optimal_parameters(self):
        """set the model's parameters as the optimal parameters"""
        self.A.data = self.opt_A.data
        self.b.data = self.opt_b.data
        self.mu.data = self.opt_mu.data

    def perturbate_weights(self, noise_scale):
        """perturbate the model's parameters with Gaussian noise"""
        self.mu.data += noise_scale * torch.randn_like(self.mu.data)
        self.A.data += noise_scale * torch.randn_like(self.A.data)
        self.b.data += noise_scale * torch.randn_like(self.b.data)

    def generate(self, z):
        """generate p(x|z)"""
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

        # q(z | x) = N(z | Ax+b, 2/3 I)
        qz, meta = self.infer(x, tau=tau)
        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        # p(z) = N(z | mu, I)
        pz = self.prior_dist(logits=self.mu)

        # compute p(x | z) = N(x | z, I)
        px = self.generate(z)

        # MSE between the parameters and the optimal paramters
        diagnostics = self._get_diagnostics()

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], **meta, **diagnostics}

    def sample_from_prior(self, N, mu=None):
        if mu is None:
            mu = self.mu
        else:
            mu = mu.view_as(self.mu)
        prior = mu.expand(N, *self.xdim)
        z = self.prior_dist(logits=prior).sample()
        px = self.generate(z)
        return {'px': px, 'z': z}

    @torch.no_grad()
    def _get_diagnostics(self):
        return {'mse_A': (self.A- self.opt_A).norm(p=2),
                'mse_b': (self.b- self.opt_b).norm(p=2),
                'mse_mu': (self.mu- self.opt_mu).norm(p=2)}
