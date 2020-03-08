from copy import copy

from torch import nn, Tensor
from torch.distributions import Bernoulli, Distribution
from torch.nn.functional import gumbel_softmax, log_softmax

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


class MLP(nn.Module):

    def __init__(self, ninp, nhid, nout, nlayers=1, bias=True, act_in=False, normalization='layernorm'):
        super().__init__()
        Norm = {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm, 'none': None, None: None}[normalization]
        layers = []
        if act_in:
            layers += [Norm(ninp), nn.ELU()]
        h = ninp
        for i in range(nlayers):
            layers += [nn.Linear(h, nhid, bias=bias)]
            if Norm is not None:
                layers += [Norm(nhid)]
            layers += [nn.ELU()]
            h = nhid
        layers += [nn.Linear(h, nout, bias=bias)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def H(z, dim=-1):
    index = z.max(dim, keepdim=True)[1]
    return torch.zeros_like(z).scatter_(dim, index, 1.0)


class Template(nn.Module):
    """A template to follow to make your model compatible with the estimators and with the training loop"""

    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """dummy forward pass"""

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
        z = qz.rsample()

        # p(x|z)
        decoder = nn.Linear(N * K, prod(dims))
        px_logits = decoder(flatten(z)).view(bs, *dims)
        px = Bernoulli(logits=px_logits)

        # values from the stochastic layers (z, pz, qz) are returned
        # as a list where each index correspond to one stochastic layer
        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': qlogits}

    def sample_from_prior(self, bs: int, **kwargs):
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

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': qlogits}


class VAE(Template):
    """
    A simple VAE model with a Categorical prior.
    """

    def __init__(self, xdim, N, K, hdim, kdim=0, nlayers=0, learn_prior=False, bias=True, normalization='layernorm',
                 likelihood=Bernoulli):
        super().__init__()
        self.xdim = xdim
        self.zdim = (N, K,)
        self.N = N
        self.K = K
        self.kdim = kdim
        args = {'nlayers': nlayers, 'bias': bias, 'normalization': normalization}

        # prior
        prior = torch.zeros((1, *self.zdim))
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
            self.qdim = self.zdim
            self.keys = None

        # encoder
        self.encoder = MLP(prod(xdim), hdim, prod(self.qdim), **args)

        # decoder
        self.decoder = MLP(prod(self.zdim), hdim, prod(xdim), **args)

        self.likelihood = likelihood

        # decoder weights
        self.decoder_weights = [v for k, v in self.decoder.named_parameters() if 'weight' in k]

    def infer(self, x):
        x = flatten(x)
        logits = self.encoder(x).view(-1, *self.qdim)

        if self.keys is not None:
            keys = self.keys / (1e-8 + self.keys.norm(dim=-1, p=2, keepdim=True) ** 2)
            logits = torch.einsum("bnh, nkh -> bnk", [logits, keys])

        return logits

    def generate(self, z):
        z = flatten(z)
        px_logits = self.decoder(z).view(-1, *self.xdim)
        return self.likelihood(logits=px_logits)

    def lipschitz(self):
        l = 1
        for w in self.decoder_weights:
            l *= w.abs().max()
        return l

    def forward(self, x, tau=0, zgrads=False, **kwargs):
        qlogits = self.infer(x)

        qz = PseudoCategorical(logits=qlogits, tau=tau)
        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = PseudoCategorical(logits=self.prior)

        px = self.generate(z)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz]}

    def sample_from_prior(self, N):
        prior = self.prior.expand(N, *self.zdim)
        z = PseudoCategorical(logits=prior).sample()
        px = self.generate(z)
        return {'px': px, 'z': z}


class Stage(nn.Module):
    def __init__(self, ninp, nhid, z, kdim, skips, bottom, top, learn_prior=False, use_baseline=True, **kwargs):
        super().__init__()

        self.skips = skips
        self.top = top
        N, K = z
        self.zdim = (N, K)

        # key-query model: log q(z|x) = Q(x)^T K
        if kdim:
            self.qdim = (N, kdim,)
            keys = torch.zeros((*self.zdim, kdim))
            self.keys = nn.Parameter(keys)
        else:
            self.qdim = self.zdim
            self.keys = None

        # deterministic inference model
        self.q_d = MLP(ninp, nhid, nhid, act_in=not bottom, **kwargs)

        # merge model q(z_i|h, z_{>i}) = MLP(z_{i>}, h)
        merge_in = nhid if top else 2 * nhid
        self.merge = MLP(merge_in, nhid, prod(self.qdim), act_in=True, **kwargs)

        # baseline model
        if use_baseline:
            _kwargs = copy(kwargs)
            _kwargs.pop('nlayers')
            self.baseline = MLP(prod(self.zdim), nhid, 1, act_in=False, **kwargs)
        else:
            self.baseline = None

        # prior
        if top:
            prior = torch.zeros((1, *self.zdim))
            if learn_prior:
                self.prior = nn.Parameter(prior)
            else:
                self.register_buffer('prior', prior)
        else:
            _kwargs = copy(kwargs)
            _kwargs.pop('nlayers')
            self.prior = MLP(nhid, nhid, prod(self.qdim), nlayers=0, act_in=True, **_kwargs)

        # generative model
        skip = nhid if skips and not top else 0
        self.v = nn.Linear(prod(self.zdim), nhid)
        self.p = MLP(nhid + skip, nhid, nhid, act_in=True, **kwargs)

    def bottom_up(self, x):
        return self.q_d(x)

    def apply_keys(self, logits):

        if self.keys is not None:
            keys = self.keys / self.keys.norm(dim=-1, p=2, keepdim=True) ** 2
            logits = torch.einsum([logits, keys], "bnh, nkh -> bnk")

        return logits

    def get_posterior(self, x_bottom, xtop=None, tau=0, **kwargs):

        h = x_bottom if xtop is None else torch.cat([x_bottom, xtop], 1)

        logits = self.merge(h)

        logits = self.apply_keys(logits)

        return PseudoCategorical(logits.view(-1, *self.zdim), tau=tau)

    def get_baseline(self, logits):
        logits = logits.detach()
        logits = flatten(logits)
        return self.baseline(logits).squeeze(1)

    def get_prior(self, xtop, N=None, tau=0, **kwargs):

        if xtop is None:
            logits = self.prior.expand(N, *self.zdim)
        else:
            logits = self.prior(xtop)
            logits = self.apply_keys(logits)

        return PseudoCategorical(logits.view(-1, *self.zdim), tau=tau)

    def top_down(self, z, skip=None):

        x = self.v(flatten(z))

        if skip is not None:
            x = torch.cat([x, skip], 1)

        return self.p(x)


class LVAE(Template):
    """
    A simple Ladder VAE model with a Categorical prior.

    /!\ changed normalization to batch norm
    """

    def __init__(self, xdim, latents, hdim, kdim=0, nlayers=0, skips=True, learn_prior=False, bias=True,
                 normalization='batchnorm', likelihood=Bernoulli, use_baseline=True, ):
        super().__init__()

        self.xdim = xdim
        self.kdim = kdim
        self.skips = skips
        self.likelihood = likelihood
        args = {'nlayers': nlayers, 'bias': bias, 'normalization': normalization,
                'learn_prior': learn_prior, 'use_baseline': use_baseline, }

        stages = []
        h = prod(xdim)
        for i, z in enumerate(latents):
            bottom = i == 0
            top = i == len(latents) - 1
            stages += [Stage(h, hdim, z, kdim, skips, bottom, top, **args)]
            h = hdim

        self.stages = nn.ModuleList(stages)

        _args = {'bias': bias, 'normalization': normalization}
        self.out = MLP(hdim, hdim, prod(self.xdim), nlayers=0, act_in=True, **_args)

    def bottom_up(self, h):

        h = flatten(h)
        for stage in self.stages:
            h = stage.bottom_up(h)
            yield h

    def top_down(self, activations, N=None, zgrads=False, **kwargs):

        if activations is None:
            activations = [None for _ in self.stages]

        output_data = DataCollector()
        xtop = None
        for stage, h in list(zip(self.stages, activations))[::-1]:

            pz = stage.get_prior(xtop, N=N, **kwargs)

            if h is not None:
                qz = stage.get_posterior(h, xtop=xtop, **kwargs)
                z = qz.rsample()
            else:
                qz = None
                z = pz.rsample()

            if not zgrads:
                z = z.detach()

            # deterministic top-down
            xtop = stage.top_down(z, skip=xtop)

            if stage.baseline is not None and qz is not None:
                baseline = stage.get_baseline(qz.logits)
            else:
                baseline = None

            output_data.extend({'z': [z], 'qz': [qz], 'pz': [pz], 'baseline': [baseline]})

        # sort data
        output_data = output_data.sort()

        xtop = self.out(xtop)
        p_x_z = self.likelihood(logits=xtop.view(-1, *self.xdim))

        return {'px': p_x_z, **output_data}

    def forward(self, x, **kwargs):
        q_activations = list(self.bottom_up(x))
        return self.top_down(activations=q_activations, N=x.size(0), **kwargs)

    def sample_from_prior(self, N, **kwargs):
        return self.top_down(None, N=N, **kwargs)
