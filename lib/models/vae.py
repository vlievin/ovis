from typing import *

import torch
from torch import nn
from torch.distributions import Distribution, Bernoulli

from .base import Template
from .layers import MLP, ConvEncoder, ConvDecoder
from ..distributions import PseudoCategorical, PseudoBernoulli, NormalFromLogits
from ..utils import batch_reduce, prod, flatten


class BaseVAE(Template):
    """
    A base VAE class with a Categorical or Gaussian prior.
    The methods `encode` and `generate` have to be implemented.
    """

    def __init__(self, xdim: Tuple[int] = tuple(),
                 N: int = 16,
                 K: int = 8,
                 hdim: int = 128,
                 kdim: int = 0,
                 dropout: float = 0.,
                 nlayers: int = 0,
                 learn_prior: bool = False,
                 bias: bool = True,
                 normalization: str = 'layernorm',
                 likelihood: Distribution = Bernoulli,
                 prior: str = 'categorical',
                 **kwargs):
        """
        Initialize a one layer VAE model.

        :param xdim: dimension of the input tensor (e.g. MNIST : xdim = (1, 28, 28))
        :param N: number of independent latent variables
        :param K: number of categories for each latent variable when using a categorical prior
        :param hdim: hidden dimension of the MLPS/ConvNets
        :param kdim: dimension of the key model for the prior of the categorical prior
        :param dropout: dropout rate
        :param nlayers: number of layers in each MLP
        :param learn_prior: learn the prior whne using a categorical prior
        :param bias: use biases in the MLPS/ConvNets
        :param normalization: normalization to be used in the MLP ['none', 'batchnorm', 'layernorm'],
        :param likelihood: distribution familly for the likelihood model p(x|z)
        :param prior: distribution familly for the prior ['normal', 'categorical']
        """
        super().__init__()

        # define prior family
        self.prior_dist = {'bernoulli': PseudoBernoulli,
                           'categorical': PseudoCategorical,
                           'normal': NormalFromLogits
                           }[prior]

        if prior == 'normal':
            K = 2
            kdim = 0
        elif prior == 'bernoulli':
            K = 1
            kdim = 0

        # output distribution p(x|z)
        self.likelihood = likelihood

        # parameters
        self.xdim = xdim
        self.zdim = (N, 1,) if prior == 'normal' else (N, K,)
        self.prior_dim = (N, 2,) if prior == 'normal' else (N, K,)
        self.N = N
        self.K = K
        self.hdim = hdim
        self.kdim = kdim
        self.dropout = dropout
        self.nlayers = nlayers
        self.bias = bias
        self.normalization = normalization

        # prior
        prior = torch.zeros((1, *self.prior_dim))
        if learn_prior:
            self.prior = nn.Parameter(prior)
        else:
            self.register_buffer('prior', prior)

        # key-query model: log q(z|x) = Q(x)^T K
        if kdim:
            self.qdim = (N, kdim,)
            keys = torch.zeros((*self.zdim, kdim)).normal_()
            self.keys = nn.Parameter(keys)
        else:
            self.qdim = (N, K)
            self.keys = None

    def encode(self, x):
        raise NotImplementedError

    def generate(self, z):
        raise NotImplementedError

    def get_logits(self, x):
        logits = self.encode(x)

        if self.keys is not None:
            keys = self.keys / (1e-8 + self.keys.norm(dim=-1, p=2, keepdim=True) ** 2)
            logits = torch.einsum("bnh, nkh -> bnk", [logits, keys])

        # important: this is required for the gradients analysis
        logits.retain_grad()

        return logits

    def infer(self, x, tau=0, mc=1, iw=1):
        """
        infer the approximate posterior q(z|x)

        :param x: observation
        :param tau: temperature parameter when using relaxation-based methods (e.g. Gumbel-Softmax)
        :param mc: number of Monte-Carlo samples
        :param iw: number of Importance-Weighted samples
        :return: posterior distribution, meta data that will be passed in the output
        """

        qlogits = self.get_logits(x)

        # we need this here so qz has the attribute .logits as `qlogits_expanded`
        # this easier for evaluation, at least for now.
        # todo: refactor to do the expansion in the sample() method
        if mc > 1 or iw > 1:
            bs, *dims = qlogits.shape
            qlogits_expanded = qlogits[:, None, None, :].expand(x.size(0), mc, iw, *dims).contiguous()
            qlogits_expanded = qlogits_expanded.view(-1, *dims)

        else:
            qlogits_expanded = qlogits

        qz = self.prior_dist(logits=qlogits_expanded, tau=tau)
        meta = {'qlogits': [qlogits]}

        return qz, meta

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs):
        """
        Compute the posterior q(z|x), sample z~q(z|x) and compute p(x|z).

        :param x: observation
        :param tau: temperature parameter when using relaxation-based methods (e.g. Gumbel-Softmax)
        :param zgrads: allow gradients through the sample z (reparametrization trick)
        :param mc: number of Monte-Carlo samples
        :param iw: number of importance weighted samples
        :param kwargs: additional keyword arguments
        :return: {'px' p(x|z), 'z': latent samples, 'qz': q(z|x), 'pz': p(z), ** additional data}:
        """

        qz, meta = self.infer(x, tau=tau, mc=mc, iw=iw)

        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = self.prior_dist(logits=self.prior)

        px = self.generate(z)

        diagnostics = self._get_diagnostics(z, qz, pz)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], **meta, **diagnostics}

    def sample_from_prior(self, N):
        """
        Sample the prior z ~ p(z) and return p(x|z)
        :param N: number of samples
        :return:
        """
        prior = self.prior.expand(N, *self.prior_dim)
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


class VAE(BaseVAE):
    """
    A simple VAE model parametrized by MLPs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # arguments for the MLPs
        args = {'nlayers': self.nlayers,
                'bias': self.bias,
                'normalization': self.normalization,
                'dropout': self.dropout}

        # encoder
        self.encoder = MLP(prod(self.xdim), self.hdim, prod(self.qdim), **args)

        # decoder
        self.decoder = MLP(prod(self.zdim), self.hdim, prod(self.xdim), **args)

    def encode(self, x):
        x = flatten(x)
        return self.encoder(x).view(-1, *self.qdim)

    def generate(self, z):
        z = flatten(z)
        px_logits = self.decoder(z).view(-1, *self.xdim)
        return self.likelihood(logits=px_logits)


class ConvVAE(BaseVAE):
    """
    A simple VAE model parametrized by convolutions
    """

    def __init__(self, padded_shape=(32, 32), **kwargs):
        super().__init__(**kwargs)

        # arguments for the ConvNets
        args = {'nlayers': self.nlayers,
                'bias': self.bias,
                'normalization': self.normalization}

        # pad input to this shape
        self.padded_shape = padded_shape
        xpad = (self.padded_shape[0] - self.xdim[1])
        ypad = self.padded_shape[1] - self.xdim[2]
        self.padding = (ypad // 2, ypad // 2, xpad // 2, xpad // 2)

        # in shape
        x_padded = [self.xdim[0], *self.padded_shape]

        # encoder
        self.conv_encoder = ConvEncoder(x_padded, self.hdim, self.hdim, **args)
        self.mlp_encoder = MLP(prod(self.conv_encoder.output_shape), 4 * self.hdim, prod(self.qdim), act_in=True,
                               **args)

        # decoder
        self.mlp_decoder = MLP(prod(self.zdim), 4 * self.hdim, prod(self.conv_encoder.output_shape), **args)
        self.conv_decoder = ConvDecoder(self.conv_encoder.output_shape, self.hdim, self.xdim[0], act_in=True, **args)

    def encode(self, x):
        # pad
        x = torch.nn.functional.pad(x, self.padding)

        y = self.conv_encoder(x)
        y = self.mlp_encoder(flatten(y))
        return y.view(-1, *self.qdim)

    def generate(self, z):
        z = flatten(z)
        x = self.mlp_decoder(z)
        px_logits = self.conv_decoder(x.view(-1, *self.conv_encoder.output_shape))
        # unpad
        px_logits = torch.nn.functional.pad(px_logits, [-p for p in self.padding])
        return self.likelihood(logits=px_logits)
