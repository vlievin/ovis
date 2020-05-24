from collections import defaultdict
from itertools import chain
from typing import *

from .base import *
from .layers import MLP
from ..distributions import Distribution, PseudoBernoulli, NormalFromLogits, PseudoCategorical


class DataCollector(defaultdict):
    def __init__(self):
        super().__init__(list)

    def extend(self, data: Dict[str, List[Optional[torch.Tensor]]]) -> None:
        """Append new data item"""
        for key, d in data.items():
            self[key] += d

    def revert(self) -> Dict[str, List[Optional[torch.Tensor]]]:
        """revert lists and return"""
        for key, d in self.items():
            self[key] = d[::-1]

        return self


class HierarchicalVae(Template):
    """
    A hierarchical VAE parametrized by the parameters \theta and \phi with L layers:

    * generative model: p_theta(x | z) p_theta(z)
    * prior: p_theta(z) = p_theta(z_L) \prod_{i=1}^{L-1} p_theta(z_i | z_{i+1})
    * inference model: q_phi(z_1 | x) \prod_{i=1}^{L-1} q_phi(z_{i+1} | z_{i})
    """

    def __init__(self, xdim: Tuple[int] = tuple(),
                 N: int = 16,
                 K: int = 8,
                 hdim: int = 128,
                 kdim: int = 0,
                 dropout: float = 0.,
                 depth: int = 3,
                 nlayers: int = 0,
                 learn_prior: bool = True,
                 bias: bool = True,
                 normalization: str = 'none',
                 likelihood: Distribution = Bernoulli,
                 x_mean: Tensor = 0.,
                 prior: str = 'categorical',
                 skip: bool = False, ):
        """
        Initialize a one layer VAE model.

        :param xdim: dimension of the input tensor (e.g. MNIST : xdim = (1, 28, 28))
        :param N: number of independent latent variables for each layer
        :param K: number of categories for each latent variable when using a categorical prior
        :param hdim: hidden dimension of the MLPS/ConvNets
        :param kdim: dimension of the key model for the prior of the categorical prior
        :param dropout: dropout rate
        :param depth: number of stochastic layers: number of layers in each MLP
        :param nlayers: number of layers in each MLP
        :param learn_prior: learn the prior whne using a categorical prior
        :param bias: use biases in the MLPS/ConvNets
        :param normalization: normalization to be used in the MLP ['none', 'batchnorm', 'layernorm'],
        :param likelihood: distribution familly for the likelihood model p(x|z)
        :param prior: distribution familly for the prior ['normal', 'categorical']
        """
        super().__init__()

        # define prior family p_{\theta}(z | * )
        self.prior_dist = {'bernoulli': PseudoBernoulli,
                           'categorical': PseudoCategorical,
                           'normal': NormalFromLogits
                           }[prior]

        # special cases depending on the choice of the prior
        # z_post is a function applied to the latent sample `z` before feeding it to the next layer
        if prior == 'normal':
            K = 2
            kdim = 0
            self.z_post = None
        elif prior == 'bernoulli':
            K = 1
            kdim = 0
            self.z_post = lambda x: 2 * x - 1
        else:
            self.z_post = None

        # output distribution p(x|z)
        self.likelihood = likelihood

        # input preprocessing
        self.register_buffer("x_mean", x_mean if isinstance(x_mean, Tensor) else 0.5 * torch.ones((1, *xdim)))
        self.x_pre = lambda x: (x - self.x_mean + 1) / 2  # function applied to the input `x`
        self.x_post = lambda x: x  # function applied to the output

        # parameters
        self.xdim = xdim  # dimension of the observation
        self.zdim = (N, 1,) if prior == 'normal' else (N, K,)  # dimension of the latent samples
        self.prior_dim = (N, 2,) if prior == 'normal' else (
        N, K,)  # dimension of the paramters of the prior distribution
        self.N = N  # number of idependent latent variables for each layer
        self.K = K  # number of categories when using categorical priors
        self.hdim = hdim  # hidden dimension used in the MLPs
        self.kdim = kdim  # key dimension when using a categorical prior with a key/query model
        self.dropout = dropout
        self.nlayers = nlayers
        self.bias = bias
        self.normalization = normalization
        self.skip = skip

        # define the parameters of the prior
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

        # arguments for all MLPs
        mlp_args = {'nlayers': nlayers,
                    'bias': bias,
                    'normalization': normalization,
                    'dropout': dropout}

        # define the inference network
        q_stages = []
        h = prod(xdim)
        for l in range(depth):
            out = prod(self.qdim)
            stage = MLP(h, hdim, out, **mlp_args)
            h = prod(self.zdim)
            if skip:
                h += out
            q_stages += [stage]
        self.encoder = nn.ModuleList(q_stages)

        # define the generative model
        p_stages = []
        hskip = 0
        for l in range(depth):
            nout = prod(self.qdim) if l < depth - 1 else prod(xdim)
            ninp = prod(self.zdim) + hskip
            stage = MLP(ninp, hdim, nout, **mlp_args)
            if skip:
                hskip = nout
            p_stages += [stage]
        self.decoder = nn.ModuleList(p_stages)

        # https://github.com/vmasrani/tvo/blob/f7a3229d954274e1d920bf4fe98dcb18f837f825/discrete_vae/models.py:
        # ``https://github.com/duvenaud/relax/blob/master/binary_vae_multilayer_per_layer.py#L273
        # https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py#L49
        # this returns the logit function (inverse of sigmoid) of clamped
        # self.train_obs_mean (see https://en.wikipedia.org/wiki/Logit)``
        self.decoder[-1].layers[-1].bias.data = - torch.log(
            1 / torch.clamp(self.x_mean.view(-1), 1e-4, 1 - 1e-4) - 1)

    def phi(self):
        return self.encoder.parameters()

    def theta(self):
        return chain(self.decoder.parameters(), self.prior)

    def encode(self, x, tau=0, zgrads=False) -> DataCollector:
        h_prev = x
        out = DataCollector()
        for layer in self.encoder:
            h = layer(h_prev)
            # q(z|h)
            qlogits = self.get_logits(h)
            qz = self.prior_dist(logits=qlogits, tau=tau)
            # z ~ q(z|h)
            z_l = qz.rsample()
            # detach z if the reparametrization trick is not used
            if not zgrads:
                z_l = z_l.detach()

            h_prev = flatten(z_l)
            if self.z_post is not None:
                h_prev = self.z_post(h_prev)

            # skip connection
            if self.skip:
                h_prev = torch.cat([h, h_prev], 1)

            out.extend({'qz': [qz], 'z': [z_l], 'qlogits': [qlogits]})

        # reorder data as L..1
        out.revert()

        return out

    def generate(self, z, tau=0, zgrads=False, N=None) -> Tuple[Distribution, DataCollector]:

        # create z as [None, ..., None] if z is None
        if z is None:
            z = [None for _ in self.decoder]
            bs = N
        else:
            bs = z[0].size(0)

        # generative process
        assert len(z) == len(self.decoder)
        h_prev = None
        out = DataCollector()
        for l, (z_l, layer) in enumerate(zip(z, self.decoder)):
            out_l = {}
            if l == 0:
                plogits = self.prior.expand(bs, *self.prior_dim)
            else:
                plogits = self.get_logits(h_prev)
            pz = self.prior_dist(logits=plogits, tau=tau)

            if z_l is None:
                z_l = pz.rsample()
                if not zgrads:
                    z_l = z_l.detach()
                out_l['z'] = [z_l]

            # MLP
            h = flatten(z_l)
            if self.z_post is not None:
                h = self.z_post(h)

            if self.skip and h_prev is not None:
                h = torch.cat([h, h_prev], 1)

            h_prev = layer(h)
            out.extend({'pz': [pz], 'plogits': [plogits], **out_l})

        px_logits = h_prev.view(-1, *self.xdim)
        px_logits = self.x_post(px_logits)
        px = self.likelihood(logits=px_logits)
        return px, out

    def get_logits(self, h):

        h = h.view(-1, *self.qdim)

        if self.keys is not None:
            keys = self.keys / (1e-8 + self.keys.norm(dim=-1, p=2, keepdim=True) ** 2)
            h = torch.einsum("bnh, nkh -> bnk", [h, keys])

        # important: this is required for the gradients analysis
        h.retain_grad()

        return h

    def infer(self, x, **kwargs):
        """
        infer the approximate posterior q(z|x)

        :param x: observation
        :param tau: temperature parameter when using relaxation-based methods (e.g. Gumbel-Softmax)
        :return: posterior distribution, meta data that will be passed in the output
        """
        x = self.x_pre(x)
        x = flatten(x)

        return self.encode(x, **kwargs)

    def forward(self, x, tau=0, zgrads=False, **kwargs):
        """
        Compute the posterior q(z|x), sample z~q(z|x) and compute p(x|z).

        :param x: observation
        :param tau: temperature parameter when using relaxation-based methods (e.g. Gumbel-Softmax)
        :param zgrads: allow gradients through the sample z (reparametrization trick)
        :param kwargs: additional keyword arguments
        :return: {'px' p(x|z), 'z': latent samples, 'qz': q(z|x), 'pz': p(z), ** additional data}:
        """

        q_data = self.infer(x, tau=tau, zgrads=zgrads)

        px, p_data = self.generate(q_data['z'], tau=tau, zgrads=zgrads)

        return {'px': px, **q_data, **p_data}

    def sample_from_prior(self, N, **kwargs):
        """
        Sample the prior z ~ p(z) and return p(x|z)
        :param N: number of samples
        :return:
        """
        px, p_data = self.generate(None, N=N, **kwargs)
        return {'px': px, **p_data}
