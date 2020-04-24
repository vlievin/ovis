from copy import copy

from torch.distributions import Bernoulli

from .distributions import *
from .layers import *
from .utils import *


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


class BaseVAE(Template):
    """
    A base VAE class with a Categorical or Gaussian prior.
    The methods `encode` and `generate` have to be implemented.
    """

    def __init__(self, xdim, N, K, hdim, kdim=0, nlayers=0, learn_prior=False, bias=True, normalization='layernorm',
                 likelihood=Bernoulli, prior='categorical'):
        super().__init__()

        # define prior family
        self.prior_dist = {'categorical': PseudoCategorical, 'normal': NormalFromLogits}[prior]
        if prior == 'normal':
            K = 2
            kdim = 0

        # output distribution p(x|z)
        self.likelihood = likelihood

        # parameters
        self.xdim = xdim
        self.zdim = (N, 1,) if prior == 'normal' else (N, K,)
        self.prior_dim = (N, 2,) if prior == 'normal' else (N, K,)
        self.N = N
        self.K = K
        self.kdim = kdim
        args = {'nlayers': nlayers, 'bias': bias, 'normalization': normalization}

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

        return qz, qlogits

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs):
        qz, qlogits = self.infer(x, tau=tau, mc=mc, iw=iw)

        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = self.prior_dist(logits=self.prior)

        px = self.generate(z)

        diagnostics = self._get_diagnostics(z, qz, pz)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': [qlogits], **diagnostics}

    def sample_from_prior(self, N):
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

    def __init__(self, xdim, N, K, hdim, nlayers=0, bias=True, dropout=0, normalization='layernorm', **kwargs):
        super().__init__(xdim, N, K, hdim, **kwargs)

        args = {'nlayers': nlayers, 'bias': bias, 'normalization': normalization, 'dropout': dropout}

        # encoder
        self.encoder = MLP(prod(xdim), hdim, prod(self.qdim), **args)

        # decoder
        self.decoder = MLP(prod(self.zdim), hdim, prod(xdim), **args)

    def encode(self, x):
        x = flatten(x)
        return self.encoder(x).view(-1, *self.qdim)

    def generate(self, z):
        z = flatten(z)
        px_logits = self.decoder(z).view(-1, *self.xdim)
        return self.likelihood(logits=px_logits)


class ConvVAE(BaseVAE):
    """
    A simple VAE model parametrized by by convolutions
    """

    def __init__(self, xdim, N, K, hdim, nlayers=0, bias=True, padded_shape=(32, 32), normalization='batchnorm',
                 **kwargs):
        super().__init__(xdim, N, K, hdim, **kwargs)

        args = {'nlayers': nlayers, 'bias': bias, 'normalization': normalization}

        # pad input to this shape
        self.padded_shape = padded_shape
        xpad = (self.padded_shape[0] - xdim[1])
        ypad = self.padded_shape[1] - xdim[2]
        self.padding = (ypad // 2, ypad // 2, xpad // 2, xpad // 2)

        # in shape
        x_padded = [xdim[0], *self.padded_shape]

        # encoder
        self.conv_encoder = ConvEncoder(x_padded, hdim, hdim, **args)
        self.mlp_encoder = MLP(prod(self.conv_encoder.output_shape), 4 * hdim, prod(self.qdim), act_in=True, **args)

        # decoder
        self.mlp_decoder = MLP(prod(self.zdim), 4 * hdim, prod(self.conv_encoder.output_shape), **args)
        self.conv_decoder = ConvDecoder(self.conv_encoder.output_shape, hdim, xdim[0], act_in=True, **args)

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

    def forward(self, x, tau=0, zgrads=False, mc=1, iw=1, **kwargs):

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
        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = self.prior_dist(logits=self.mu)

        px = self.generate(z)

        diagnostics = self._get_diagnostics(z, qz, pz)

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': [qlogits], 'b': [b], **diagnostics}

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

        qz, qlogits = self.infer(x, tau=tau, mc=mc, iw=iw)

        z = qz.rsample()

        if not zgrads:
            z = z.detach()

        pz = self.prior_dist(logits=self.log_theta)

        px = self.generate(z)

        diagnostics = self._get_diagnostics(x, self.prior_dist(logits=qlogits))

        return {'px': px, 'z': [z], 'qz': [qz], 'pz': [pz], 'qlogits': [qlogits], **diagnostics}

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
