from collections import defaultdict
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
                 prior: str = 'categorical'):
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

        # define prior family
        self.prior_dist = {'bernoulli': PseudoBernoulli,
                           'categorical': PseudoCategorical,
                           'normal': NormalFromLogits
                           }[prior]

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
        self.register_buffer("x_mean", x_mean if isinstance(x_mean, Tensor) else torch.tensor(0.))
        self.x_pre = lambda x: (x - self.x_mean + 1) / 2
        self.x_post = lambda x: x - torch.log(1. / torch.clamp(self.x_mean, 0.001, 0.999) - 1)


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

        mlp_args = {'nlayers': nlayers,
                    'bias': bias,
                    'normalization': normalization,
                    'dropout': dropout}

        # define the encoder
        q_stages = []
        h = prod(xdim)
        for l in range(depth):
            stage = MLP(h, hdim, prod(self.qdim), **mlp_args)
            h = prod(self.zdim)
            q_stages += [stage]
        self.encoder = nn.ModuleList(q_stages)

        # define the decoder
        p_stages = []
        for l in range(depth):
            nout = prod(self.qdim) if l < depth - 1 else prod(xdim)
            stage = MLP(prod(self.zdim), hdim, nout, **mlp_args)
            p_stages += [stage]
        self.decoder = nn.ModuleList(p_stages)

    def encode(self, x, tau=0, zgrads=False) -> DataCollector:
        h = x
        out = DataCollector()
        for layer in self.encoder:
            h = layer(h)
            # q(z|h)
            qlogits = self.get_logits(h)
            qz = self.prior_dist(logits=qlogits, tau=tau)
            # z ~ q(z|h)
            z_l = qz.rsample()
            # detach z if the reparametrization trick is not used
            if not zgrads:
                z_l = z_l.detach()

            h = flatten(z_l)
            if self.z_post is not None:
                h = self.z_post(h)
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
        h = None
        out = DataCollector()
        for l, (z_l, layer) in enumerate(zip(z, self.decoder)):
            out_l = {}
            if l == 0:
                plogits = self.prior.expand(bs, *self.prior_dim)
            else:
                plogits = self.get_logits(h)
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

            h = layer(h)
            out.extend({'pz': [pz], 'plogits': [plogits], **out_l})

        px_logits = h.view(-1, *self.xdim)
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

    def infer(self, x, mc=1, iw=1, **kwargs):
        """
        infer the approximate posterior q(z|x)

        :param x: observation
        :param tau: temperature parameter when using relaxation-based methods (e.g. Gumbel-Softmax)
        :param mc: number of Monte-Carlo samples
        :param iw: number of Importance-Weighted samples
        :return: posterior distribution, meta data that will be passed in the output
        """
        x = self.x_pre(x)
        x = flatten(x)

        if mc > 1 or iw > 1:
            bs, *dims = x.shape
            x = x[:, None, None, :].expand(x.size(0), mc, iw, *dims).contiguous()
            x = x.view(-1, *dims)

        return self.encode(x, **kwargs)

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

        q_data = self.infer(x, mc=mc, iw=iw, tau=tau, zgrads=zgrads)

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
