# Using the Sigmoid Belief Network from `The Thermodynamic Variational Objective` [https://arxiv.org/abs/1907.00031]
# original code at [https://github.com/vmasrani/tvo/blob/f7a3229d954274e1d920bf4fe98dcb18f837f825/discrete_vae/models.py]

from typing import *

import torch
from torch import nn, Tensor
from torch.distributions import Distribution

from .base import Template
from ..utils.utils import prod


class ChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist, get_next_dist):
        self.chain_dist = chain_dist
        self.get_next_dist = get_next_dist

    def sample(self, sample_shape=torch.Size()):
        sample_chain = self.chain_dist.sample(sample_shape=sample_shape)
        sample_next = self.get_next_dist(sample_chain[-1]).sample(
            sample_shape=())
        return sample_chain + (sample_next,)

    def log_prob(self, value):
        log_prob_chain = self.chain_dist.log_prob(value[:-1])
        log_prob_next = self.get_next_dist(value[-2]).log_prob(value[-1])
        return log_prob_chain + log_prob_next


class ChainDistributionFromSingle(torch.distributions.Distribution):
    def __init__(self, single_dist):
        self.single_dist = single_dist

    def sample(self, sample_shape=torch.Size()):
        return (self.single_dist.sample(sample_shape=sample_shape),)

    def log_prob(self, value):
        return self.single_dist.log_prob(value[0])


class ReversedChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist):
        self.chain_dist = chain_dist

    def sample(self, sample_shape=torch.Size()):
        return tuple(reversed(self.chain_dist.sample(
            sample_shape=sample_shape)))

    def log_prob(self, value):
        return self.chain_dist.log_prob(tuple(reversed(value)))


class MultilayerPerceptron(nn.Module):
    def __init__(self, dims, non_linearity):
        """
        Args:
            dims: list of ints
            non_linearity: differentiable function

        Returns: nn.Module which represents an MLP with architecture

            x -> Linear(dims[0], dims[1]) -> non_linearity ->
            ...
            Linear(dims[-3], dims[-2]) -> non_linearity ->
            Linear(dims[-2], dims[-1]) -> y"""

        super(MultilayerPerceptron, self).__init__()
        self.dims = dims
        self.non_linearity = non_linearity
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linear_modules.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        temp = x
        for linear_module in self.linear_modules[:-1]:
            temp = self.non_linearity(linear_module(temp))
        return self.linear_modules[-1](temp)


def init_mlp(in_dim, out_dim, hidden_dim, num_layers, non_linearity):
    """Initializes a MultilayerPerceptron.

    Args:
        in_dim: int
        out_dim: int
        hidden_dim: int
        num_layers: int
        non_linearity: differentiable function

    Returns: a MultilayerPerceptron with the architecture

        x -> Linear(in_dim, hidden_dim) -> non_linearity ->
        ...
        Linear(hidden_dim, hidden_dim) -> non_linearity ->
        Linear(hidden_dim, out_dim) -> y

        where num_layers = 0 corresponds to

        x -> Linear(in_dim, out_dim) -> y"""
    dims = [in_dim] + [hidden_dim for _ in range(num_layers)] + [out_dim]
    return MultilayerPerceptron(dims, nn.Tanh())


class GenerativeModel(nn.Module):
    def __init__(self, num_stochastic_layers=3, num_deterministic_layers=0,
                 latent_dim=200, obs_dim=784, train_obs_mean=None,
                 device=torch.device('cpu'), latent_dims=None, out_shape=(1, 28, 28), **kwargs):
        """
        Args:
            num_stochastic_layers: int; 1 corresponds to a p(z)p(x | z)
            num_deterministic_layers: int; 0 corresponds to linear layers
            latent_dim: int; default = 200
            obs_dim: int; default = 784
            train_obs_mean: tensor of shape [obs_dim]
            device: torch.device('cpu') or torch.device('cuda')
            latent_dims: if specified, overrides num_stochastic_layers and
                latent_dim; latent_dims[0] is the dimensionality of the
                latent layer that is furthest from obs"""
        super(GenerativeModel, self).__init__()
        if latent_dims is None:
            self.num_stochastic_layers = num_stochastic_layers
            self.latent_dim = latent_dim
            self.latent_dims = [latent_dim for _ in
                                range(num_stochastic_layers)]
        else:
            self.num_stochastic_layers = len(latent_dims)
            self.latent_dim = None
            self.latent_dims = latent_dims
        self.num_deterministic_layers = num_deterministic_layers
        self.obs_dim = obs_dim
        if train_obs_mean is None:
            self.register_buffer("train_obs_mean", torch.ones(
                self.obs_dim, device=device, dtype=torch.float) / 2)
        else:
            self.register_buffer("train_obs_mean", train_obs_mean)
        self.device = device
        self.latent_param_logits = nn.Parameter(
            torch.zeros(self.latent_dims[0], device=device, dtype=torch.float))
        self.decoders = nn.ModuleDict()
        for i in range(1, self.num_stochastic_layers):
            self.decoders[str(i)] = init_mlp(
                in_dim=self.latent_dims[i - 1], out_dim=self.latent_dims[i],
                hidden_dim=self.latent_dims[i - 1],
                num_layers=num_deterministic_layers, non_linearity=nn.Tanh())
        self.decoder_to_obs = init_mlp(in_dim=self.latent_dims[-1],
                                       out_dim=obs_dim,
                                       hidden_dim=self.latent_dims[-1],
                                       num_layers=num_deterministic_layers,
                                       non_linearity=nn.Tanh())

        # https://github.com/duvenaud/relax/blob/master/binary_vae_multilayer_per_layer.py#L273
        # https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py#L49
        # this returns the logit function (inverse of sigmoid) of clamped
        # self.train_obs_mean (see https://en.wikipedia.org/wiki/Logit)
        self.decoder_to_obs.linear_modules[-1].bias.data = -torch.log(
            1 / torch.clamp(self.train_obs_mean, 1e-4, 1 - 1e-4) - 1)

        self.out_shape = out_shape

    def get_latent_layer_param(self, layer_idx, previous_latent_layer=None):
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx > 0)

        Returns: logits for Bernoulli latent
            if layer_idx is 0, shape [latent_dim],
            otherwise [batch_size, latent_dim]"""
        if layer_idx == 0:
            return self.latent_param_logits
        else:
            return self.decoders[str(layer_idx)](previous_latent_layer * 2 - 1)

    def get_latent_layer_dist(self, layer_idx, previous_latent_layer=None):
        """Returns distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx > 0)

        Returns: Bernoulli distribution with event_shape [latent_dim]
            if layer_idx is 0, batch_shape is [],
            otherwise [batch_size]"""
        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                logits=self.get_latent_layer_param(layer_idx,
                                                   previous_latent_layer)),
            reinterpreted_batch_ndims=1)

    def get_latent_dist(self):
        """Returns: distribution for all latent layers:

            dist.sample(sample_shape=[sample_shape]) returns
            (latent_0, ..., latent_N) where each latent_n
            is of shape [sample_shape, latent_dim] and latent_0
            corresponds to the latent furthest away from obs

            if latent_n is of shape [batch_shape, latent_dim]
            dist.log_prob(latent_0, ..., latent_N) returns
            sum_n log_prob(latent_n) which is of shape [batch_shape]"""

        latent_dist = ChainDistributionFromSingle(
            self.get_latent_layer_dist(layer_idx=0))
        for layer_idx in range(1, self.num_stochastic_layers):
            # be careful about closures
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture/2295372
            latent_dist = ChainDistribution(
                latent_dist,
                lambda previous_latent_layer, layer_idx=layer_idx:
                self.get_latent_layer_dist(
                    layer_idx=layer_idx,
                    previous_latent_layer=previous_latent_layer))
        return latent_dist

    def get_obs_param(self, latent):
        """Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}

        Returns: logits of Bernoulli likelihood of shape
            [batch_size, obs_dim]
        """
        latent_layer = latent[-1]

        # https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py#L265
        # https://github.com/duvenaud/relax/blob/master/binary_vae_multilayer_per_layer.py#L159-L160
        logits = self.decoder_to_obs(latent_layer * 2 - 1)
        return logits.view(-1, *self.out_shape)

    def get_obs_dist(self, latent):
        """Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}

        Returns: Bernoulli distribution of batch_shape [batch_size] and
            event_shape [obs_dim]
        """
        return torch.distributions.Independent(
            # https://github.com/tensorflow/models/blob/master/research/rebar/utils.py#L93-L105
            # https://github.com/duvenaud/relax/blob/master/rebar_tf.py#L36-L37
            torch.distributions.Bernoulli(logits=self.get_obs_param(latent)),
            reinterpreted_batch_ndims=1)

    def get_log_prob(self, latent, obs):
        """Log of joint probability.

        Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}

            obs: tensor of shape [batch_size, obs_dim] with values in {0, 1}

        Returns: tensor of shape [num_particles, batch_size]
        """

        latent_log_prob = self.get_latent_dist().log_prob(latent)
        obs_log_prob = self.get_obs_dist(latent).log_prob(obs)
        return latent_log_prob + obs_log_prob

    def sample_latent_and_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}
            obs: tensor of shape [num_samples, obs_dim]
        """
        latent_dist = self.get_latent_dist()
        latent = latent_dist.sample((num_samples,))
        obs_dist = self.get_obs_dist(latent)
        obs = obs_dist.sample()

        return latent, obs

    def sample_obs(self, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            obs: tensor of shape [num_samples, obs_dim]
        """
        return self.sample_latent_and_obs(num_samples)[1]


class InferenceNetwork(Template):
    def __init__(self, num_stochastic_layers=3, num_deterministic_layers=0,
                 latent_dim=200, obs_dim=784, train_obs_mean=None,
                 device=torch.device('cpu'), latent_dims=None, **kwargs):
        """Args:
            num_stochastic_layers: int; 1 corresponds to a p(z)p(x | z)
            num_deterministic_layers: int; 0 corresponds to linear layers
            latent_dim: int; default = 200
            obs_dim: int; default = 784
            train_obs_mean: tensor of shape [obs_dim]
            device: torch.device('cpu') or torch.device('cuda')
            latent_dims: if specified, overrides num_stochastic_layers and
                latent_dim; latent_dims[0] is the dimensionality of the
                latent layer that is furthest from obs
        """
        super(InferenceNetwork, self).__init__()
        if latent_dims is None:
            self.num_stochastic_layers = num_stochastic_layers
            self.latent_dim = latent_dim
            self.latent_dims = [latent_dim for _ in
                                range(num_stochastic_layers)]
        else:
            self.num_stochastic_layers = len(latent_dims)
            self.latent_dim = None
            self.latent_dims = latent_dims
        self.num_deterministic_layers = num_deterministic_layers
        self.obs_dim = obs_dim
        if train_obs_mean is None:
            self.register_buffer("train_obs_mean", torch.ones(
                self.obs_dim, device=device, dtype=torch.float) / 2)
        else:
            self.register_buffer("train_obs_mean", train_obs_mean)
        self.device = device

        self.encoder_to_obs = init_mlp(in_dim=obs_dim,
                                       out_dim=self.latent_dims[-1],
                                       hidden_dim=self.latent_dims[-1],
                                       num_layers=num_deterministic_layers,
                                       non_linearity=nn.Tanh())
        self.encoders = nn.ModuleDict()
        for i in reversed(range(self.num_stochastic_layers - 1)):
            self.encoders[str(i)] = init_mlp(
                in_dim=self.latent_dims[i + 1], out_dim=self.latent_dims[i],
                hidden_dim=self.latent_dims[i + 1],
                num_layers=num_deterministic_layers, non_linearity=nn.Tanh())

        if train_obs_mean is None:
            self.train_obs_mean = torch.ones(
                obs_dim, device=device, dtype=torch.float) / 2
        else:
            self.train_obs_mean = train_obs_mean

    def get_latent_layer_param(self, layer_idx, previous_latent_layer=None,
                               obs=None):
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [batch_size, obs_dim] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns: logits for Bernoulli latent of shape
            [batch_size, latent_dim]"""
        if layer_idx == self.num_stochastic_layers - 1:
            # https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py#L169-L172
            # https://github.com/duvenaud/relax/blob/master/binary_vae_multilayer_per_layer.py#L127
            return self.encoder_to_obs((obs - self.train_obs_mean + 1) / 2)
        else:
            return self.encoders[str(layer_idx)](previous_latent_layer * 2 - 1)

    def get_latent_layer_dist(self, layer_idx, previous_latent_layer=None,
                              obs=None):
        """Returns distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [batch_size, obs_dim] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns: Bernoulli distribution with event_shape [latent_dim]
            and batch_shape is [batch_size]"""
        return torch.distributions.Independent(
            torch.distributions.Bernoulli(
                logits=self.get_latent_layer_param(
                    layer_idx, previous_latent_layer, obs)),
            reinterpreted_batch_ndims=1)

    def get_latent_dist(self, obs):
        """Args:
            obs: tensor of shape [batch_size, obs_dim] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns: distribution for all latent layers:

            dist.sample(sample_shape=[sample_shape]) returns
            (latent_0, ..., latent_N) where each latent_n
            is of shape [sample_shape, latent_dim] and latent_0
            corresponds to the latent furthest away from obs

            if latent_n is of shape [batch_shape, latent_dim]
            dist.log_prob(latent_0, ..., latent_N) returns
            sum_n log_prob(latent_n) which is of shape [batch_shape]"""

        latent_dist = ChainDistributionFromSingle(
            self.get_latent_layer_dist(
                layer_idx=self.num_stochastic_layers - 1, obs=obs))
        for layer_idx in reversed(range(self.num_stochastic_layers - 1)):
            # be careful about closures
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture/2295372
            latent_dist = ChainDistribution(
                latent_dist,
                lambda previous_latent_layer, layer_idx=layer_idx:
                self.get_latent_layer_dist(
                    layer_idx=layer_idx,
                    previous_latent_layer=previous_latent_layer))
        return ReversedChainDistribution(latent_dist)

    def sample_from_latent_dist(self, latent_dist, num_samples=None):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [latent_dim]
            num_samples: int

        Returns:
            latent: tensor of shape [num_samples, batch_size, num_mixtures]
        """
        sample_shape = (num_samples,) if num_samples is not None else tuple()
        return latent_dist.sample(sample_shape)

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(latent | obs).

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [latent_dim]
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}

        Returns: tensor of shape [batch_size]
        """
        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs):
        """Log q(latent | obs).

        Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}
            obs: tensor of shape [batch_size, obs_dim]

        Returns: tensor of shape [batch_size]
        """
        return self.get_log_prob_from_latent_dist(
            self.get_latent_dist(obs), latent)


def init_models(train_obs_mean, architecture, device):
    """Args:
        train_obs_mean: tensor of shape [obs_dim]
        architecture: linear_1, linear_2, linear_3 or non_linear
        device: torch.device

    Returns: generative_model, inference_network
    """

    if architecture[:len('linear')] == 'linear':
        num_stochastic_layers = int(architecture[-1])
        generative_model = GenerativeModel(
            num_stochastic_layers=num_stochastic_layers,
            num_deterministic_layers=0,
            device=device, train_obs_mean=train_obs_mean)
        inference_network = InferenceNetwork(
            num_stochastic_layers=num_stochastic_layers,
            num_deterministic_layers=0,
            device=device, train_obs_mean=train_obs_mean)
    elif architecture == 'non_linear':
        generative_model = GenerativeModel(
            num_stochastic_layers=1,
            num_deterministic_layers=2,
            device=device, train_obs_mean=train_obs_mean)
        inference_network = InferenceNetwork(
            num_stochastic_layers=1,
            num_deterministic_layers=2,
            device=device, train_obs_mean=train_obs_mean)

    if device.type == 'cuda':
        generative_model.cuda()
        inference_network.cuda()

    return generative_model, inference_network


class SigmoidBeliefNetwork(nn.Module):
    """A wrapper for the official TVO model."""

    def __init__(self, xdim: Tuple[int] = tuple(),
                 N: int = 200,
                 depth: int = 3,
                 nlayers: int = 0,
                 x_mean: Optional[Tensor] = None,
                 **kwargs):
        super().__init__()

        obs_dim = prod(xdim)
        args = {'obs_dim': obs_dim, 'latent_dim': N, 'out_shape': xdim}

        self.generative_model = GenerativeModel(
            num_stochastic_layers=depth,
            num_deterministic_layers=nlayers,
            train_obs_mean=x_mean.view(-1),
            **args)

        self.inference_network = InferenceNetwork(
            num_stochastic_layers=depth,
            num_deterministic_layers=nlayers,
            train_obs_mean=x_mean.view(-1),
            **args)

    def forward(self, x, **kwargs):
        x = x.view(x.size(0), -1)

        # compute the distribution q(z|x)
        qz = self.inference_network.get_latent_dist(x)
        # sample z ~ q(z|x)
        z = self.inference_network.sample_from_latent_dist(qz)

        # compute the distribution p(z)
        pz = self.generative_model.get_latent_dist()

        # compute the distribution p(x|z)
        px = self.generative_model.get_obs_dist(z)

        return {'z': [z], 'pz': [pz], 'qz': [qz], 'px': px}

    def theta(self):
        return self.generative_model.parameters()

    def phi(self):
        return self.inference_network.parameters()

    def sample_from_prior(self, N, **kwargs):
        # compute the distribution p(z)
        latent_dist = self.generative_model.get_latent_dist()
        z = latent_dist.sample((N,))
        px = self.generative_model.get_obs_dist(z)

        return {'px': px, 'z': z}
