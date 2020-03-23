import numpy as np
from booster import Diagnostic
from torch import nn, Tensor
from torch.nn.functional import softmax

from .baseline import Baseline
from .model import PseudoCategorical, VAE
from .utils import *

_EPS = 1e-18


class Estimator(nn.Module):
    def __init__(self, beta: float = 1, mc: int = 1, iw: int = 1, sequential_computation=False, freebits=0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mc = mc
        self.iw = iw
        self.log_iw = np.log(iw)
        self.log_mc = np.log(mc)
        self.log_mc_iw_m1 = np.log(mc * iw - 1)
        self.freebits = FreeBits(freebits)
        self.detach_qlogits = False
        self.sequential_computation = sequential_computation

    def _expand_sample(self, x):
        bs, *dims = x.size()
        self.bs = bs  # added for TVO - perhaps a more elegant fix is possible.
        x = x[:, None, None].repeat(1, self.mc, self.iw, *(1 for _ in dims))
        # flatten everything into the batch dimension
        return x.view(-1, *dims)

    def _reduce_sample(self, x):
        _, *dims = x.size()
        x = x.view(-1, self.mc, self.iw, *dims)
        return x.mean((1, 2,))

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs) -> Tuple[Tensor, Dict, Dict]:
        """
        Compute the loss given the `model` and a batch of data `x`. Returns the loss per datapoint, diagnostics and the model's output
        :param model: nn.Module
        :param x: batch of data
        :param backward: compute backward pass
        :param kwargs:
        :return: loss, diagnostics, model's output
        """
        raise NotImplementedError


class ThermoVariationalObjective(Estimator):
    """
    Thermovariational Inference. Based on https://arxiv.org/pdf/1907.00031.pdf / https://github.com/vmasrani/tvo
    Currently implemented for discrete models.

    TODO: Continuous models?
    """

    def compute_loss(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor], partition=1,
                     integration='left') -> Dict[str, Tensor]:
        """
        TODO: ADD PARTITION ARGUMENT

        Computes the covariance gradient estimator for the TVO bound.

         TVO = Delta_beto_0 ELBO + sum_{i=1..K-1} Delta_beta_k E_{pi_{beta_k]}}( f(x,z) )], f(x,z) = log p(x,z) - log q(z | x)

         Gradient estimator, eq. 11:
         Nabla E_{pi_{beta_k]}}( f(x,z) )] = E_{pi_{beta_k]}}( Nabla f(x,z) )] + Cov( Nabla log tilde{pi}(z), f(z) )

         (((In this expression the KLs are concatenated by stochastic layer so the freebits can be applied to each of them.)))

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
        :param num_particles: int; number of samples S in importance sampling of expectations
        :param integration: type of integral approximation (Riemann sum); left, right or trapz
        :return: dictionary with outputs [tvo, elbo, kl, log_f_x,z, N_eff]
        """

        # Generalize to accept custom partitions (current partition taken from default TVO settings on github)
        partition = torch.FloatTensor([0.0000e+00, 1.0000e-10, 1.2915e-09, 1.6681e-08, 2.1544e-07, 2.7826e-06,
                                       3.5938e-05, 4.6416e-04, 5.9948e-03, 7.7426e-02, 1.0000e+00]).cuda()

        num_particles = self.iw
        bs = self.bs  # temporary

        # compute the effective sample size
        N_eff = self.effective_sample_size(log_pzs, log_qzs)

        def cat_by_layer(log_pzs):
            """sum over the latent dimension (N_i) and concatenate stocastic layers.
            for L=2, a list of values [[*, N_1], [*, N_2]] becomes [*, 2]"""
            return torch.cat([x.sum(1, keepdims=True) for x in log_pzs], 1)

        # kl = E_q[ log p(z) - log q(z) ]
        log_pz = cat_by_layer(log_pzs)
        log_qz = cat_by_layer(log_qzs)
        if self.detach_qlogits:
            log_qz = log_qz.detach()
        kl = log_qz - log_pz

        # freebits is ditributed equally over the last dimension
        # (meaning L layers result in a total of L * freebits budget)
        kl = self.freebits(kl.unsqueeze(-1))
        kl = batch_reduce(kl)

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # compute log of weights w_s = p(x,z)/q(z|x) (between eq. 13 and 14)
        log_weight = log_f_xz.view(bs, num_particles)  # log_p_xz - log_q
        heated_log_weight = log_weight.unsqueeze(-1) * partition

        def exponentiate_and_normalize(values, dim=0):
            """Exponentiates and normalizes a tensor.
            Args:
                values: tensor [dim_1, ..., dim_N]
                dim: n
            Returns:
                result: tensor [dim_1, ..., dim_N]
                    where result[i_1, ..., i_N] =
                                    exp(values[i_1, ..., i_N])
                    ------------------------------------------------------------
                     sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N]) """
            log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
            return torch.exp(values - log_denominator)

        # compute bar{w^\beta_s} (used in eq. 13)
        heated_normalized_weight = exponentiate_and_normalize(heated_log_weight, dim=1)

        # compute tilde{pi}_beta(z) (eq. 7)
        log_p = log_px_z.view(-1, 1) + log_pz  # <---- is this correct ?
        log_p = log_p.view(bs, num_particles)
        log_qz = log_qz.view(bs, num_particles)
        thermo_logp = partition * log_p.unsqueeze(-1) + (1 - partition) * log_qz.unsqueeze(-1)

        # compute product in eq. 13 with f(x,z) = log[p(x,z)/q(z|x)]
        wf = heated_normalized_weight * log_weight.unsqueeze(-1)
        w_detached = heated_normalized_weight.detach()

        # num_particles S for importance sampling as
        # described in subsection Expectations of Section 4
        if num_particles == 1:
            correction = 1
        else:
            correction = num_particles / (num_particles - 1)

        # compute covariance (eq. 12)
        # .detach() makes sure PyTorch does not differentiate f_lambda(z) term.
        cov_term = correction * torch.sum(w_detached * (log_weight.unsqueeze(-1) -
                                                        torch.sum(wf, dim=1, keepdim=True)).detach() * (thermo_logp -
                                                                                                        torch.sum(
                                                                                                            thermo_logp * w_detached,
                                                                                                            dim=1,
                                                                                                            keepdim=True)),
                                          dim=1)

        # compute distances of partioning
        multiplier = torch.zeros_like(partition)
        if integration == 'trapz':
            multiplier[0] = 0.5 * (partition[1] - partition[0])
            multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
            multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
        elif integration == 'left':
            multiplier[:-1] = partition[1:] - partition[:-1]
        elif integration == 'right':
            multiplier[1:] = partition[1:] - partition[:-1]

        # compute covariance gradient estimator (eq. 11 / appendix F)

        """ Take mean over samples """
        # tvo = torch.mean(torch.sum(
        # multiplier * (thing_to_add + torch.sum(w_detached * log_weight.unsqueeze(-1),
        # dim=1)), dim=1))

        """ No mean over samples (necessary in this setup) """
        tvo = torch.sum(
            multiplier * (cov_term + torch.sum(w_detached * log_weight.unsqueeze(-1),
                                               dim=1)), dim=1)

        # multisampling ELBO
        log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
        elbo = torch.mean(log_evidence)

        return {'tvo': tvo, 'elbo': elbo, 'kl': kl, 'log_f_xz': log_f_xz, 'N_eff': N_eff}

    def effective_sample_size(self, log_pz, log_qz):

        # Not changed anything for TVO

        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2
        :param log_pz: log p(z) of shape [bs * mc * iw, ...]
        :param log_qz: loq q(z) of shape [bs * mc * iw, ...]
        :return: effective_sample_size
        """

        if isinstance(log_pz, List):
            log_pz = torch.cat(log_pz, 1)

        if isinstance(log_qz, List):
            log_qz = torch.cat(log_qz, 1)

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            w = batch_reduce(log_pz - log_qz).exp().view(-1, self.mc, self.iw)
            N_eff = torch.sum(w, 2) ** 2 / torch.sum(w ** 2, 2)
            N_eff = N_eff.mean(1)  # MC
        else:
            x = (log_pz).view(-1, self.mc, self.iw)
            N_eff = torch.ones_like(x[:, 0, 0])

        return N_eff

    def evaluate_model(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Dict[str, Tensor]:

        # Not changed anything for TVO

        if self.sequential_computation:
            # todo: move this in the forward pass so we retain only point estimates (loop over `evaluate model` method with iw = 1)
            bs, *dims = x.size()
            log_px_zs = []
            log_pzs = []
            log_qzs = []

            for i in range(self.iw):
                __x = x[:, None].repeat(1, self.mc, *(1 for _ in dims)).view(-1, *dims)

                # forward pass
                output = model(__x, **kwargs)
                px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

                # compute log p(x|z), log p(z) and log q(z | x)
                log_px_zs += [batch_reduce(px.log_prob(__x))]
                log_pzs += [[pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]]
                log_qzs += [[qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]]

            # concatenate
            log_px_z = torch.cat([x.view(bs, self.mc, 1) for x in log_px_zs], 2)

            log_pzs = [list(x) for x in zip(*log_pzs)]
            log_pz = [torch.cat([x.view(bs, self.mc, 1, *x.size()[1:]) for x in log_pzs_l], 2) for log_pzs_l in log_pzs]

            log_qzs = [list(x) for x in zip(*log_qzs)]
            log_qz = [torch.cat([x.view(bs, self.mc, 1, *x.size()[1:]) for x in log_qzs_l], 2) for log_qzs_l in log_qzs]

            # re-flatten bs, mc, iw samples
            log_px_z = log_px_z.view(-1)
            log_pz = [x.view(-1, *x.size()[3:]) for x in log_pz]
            log_qz = [x.view(-1, *x.size()[3:]) for x in log_qz]

            # for now this does not return the model output as it is complicated to concatenate all of them (distributions)
            # this is why this should be used only for evaluation using basic VI
            return {'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz}

        else:
            # expand input to get MC and IW samples: [bs, mc, iw, ...]
            __x = self._expand_sample(x)

            # forward pass
            output = model(__x, **kwargs)
            px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]
            # compute log p(x|z), log p(z) and log q(z | x)
            log_px_z = batch_reduce(px.log_prob(__x))
            log_pz = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
            log_qz = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

            output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})
            return output

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        # Removed '.mean(1)'s and changed namings for TVO

        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        tvo_data = self.compute_loss(log_px_z, log_pz, log_qz)

        # loss
        tvo = tvo_data.get('tvo')  # .mean(1)  # MC averaging (done in loss above - move here?)
        loss = - tvo

        elbo = tvo_data.get('elbo')  # .mean(1)  # MC averaging (done in loss above - move here?)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': elbo,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(tvo_data.get('kl')),
                     'N_eff': tvo_data.get('N_eff')},
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output


class VariationalInference(Estimator):
    """
    Variational Inference. Using this estimator requires the model to be compatible with the reparametrization trick.

    """

    def compute_iw_bound(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor],
                         detach_qlogits: bool = False) -> Dict[str, Tensor]:
        """
        Compute the importance weighted bound:

         L_k = E_{q(z_{1..K} | x)} [ log 1/K \sum_{i=1..K} f(x, z_i)], f(x, z) = p(x,z) / q(z|x)

         In this expression the KLs are concatenated by stochastic layer so the freebits can be applied to each of them.

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param detach_qlogits: detach the logits of q(z|x)
        :return: dictionary with outputs [L_k, kl, log_f_x,z]
        """

        # compute the effective sample size
        N_eff = self.effective_sample_size(log_pzs, log_qzs)

        def cat_by_layer(log_pzs):
            """sum over the latent dimension (N_i) and concatenate stocastic layers.
            for L=2, a list of values [[*, N_1], [*, N_2]] becomes [*, 2]"""
            return torch.cat([x.sum(1, keepdims=True) for x in log_pzs], 1)

        # kl = E_q[ log p(z) - log q(z) ]
        log_pz = cat_by_layer(log_pzs)
        log_qz = cat_by_layer(log_qzs)
        if detach_qlogits:
            log_qz = log_qz.detach()
        kl = log_qz - log_pz
        # freebits is ditributed equally over the last dimension
        # (meaning L layers result in a total of L * freebits budget)
        kl = self.freebits(kl.unsqueeze(-1))
        kl = batch_reduce(kl)

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(-1, self.mc, self.iw)

        # IW-ELBO: L_k
        L_k = torch.logsumexp(log_f_xz, dim=2) - self.log_iw  # if self.iw > 1 else log_f_xz.squeeze(2)

        return {'L_k': L_k, 'kl': kl, 'log_f_xz': log_f_xz, 'N_eff': N_eff}

    def effective_sample_size(self, log_pz, log_qz):
        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2
        :param log_pz: log p(z) of shape [bs * mc * iw, ...]
        :param log_qz: loq q(z) of shape [bs * mc * iw, ...]
        :return: effective_sample_size
        """

        if isinstance(log_pz, List):
            log_pz = torch.cat(log_pz, 1)

        if isinstance(log_qz, List):
            log_qz = torch.cat(log_qz, 1)

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            w = batch_reduce(log_pz - log_qz).exp().view(-1, self.mc, self.iw)
            N_eff = torch.sum(w, 2) ** 2 / torch.sum(w ** 2, 2)
            N_eff = N_eff.mean(1)  # MC
        else:
            x = (log_pz).view(-1, self.mc, self.iw)
            N_eff = torch.ones_like(x[:, 0, 0])

        return N_eff

    def evaluate_model(self, model: nn.Module, x: Tensor, x_target: Tensor, **kwargs: Any) -> Dict[str, Tensor]:
        # forward pass
        output = model(x, **kwargs)
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x_target))
        log_pz = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
        log_qz = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

        output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})
        return output

    def _sequential_evaluation(self, model: nn.Module, x: Tensor, **kwargs: Any):
        bs, *dims = x.size()
        log_px_zs = []
        log_pzs = []
        log_qzs = []
        x_target = x[:, None].repeat(1, self.mc, *(1 for _ in dims)).view(-1, *dims)
        for i in range(self.iw):
            # evaluate batch
            output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=1, **kwargs)
            log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]

            log_px_zs += [log_px_z]
            log_pzs += [log_pz]
            log_qzs += [log_qz]

            # concatenate
        log_px_z = torch.cat([x.view(bs, self.mc, 1) for x in log_px_zs], 2)

        log_pzs = [list(x) for x in zip(*log_pzs)]
        log_pz = [torch.cat([x.view(bs, self.mc, 1, *x.size()[1:]) for x in log_pzs_l], 2) for log_pzs_l in log_pzs]

        log_qzs = [list(x) for x in zip(*log_qzs)]
        log_qz = [torch.cat([x.view(bs, self.mc, 1, *x.size()[1:]) for x in log_qzs_l], 2) for log_qzs_l in log_qzs]

        # re-flatten bs, mc, iw samples
        log_px_z = log_px_z.view(-1)
        log_pz = [x.view(-1, *x.size()[3:]) for x in log_pz]
        log_qz = [x.view(-1, *x.size()[3:]) for x in log_qz]

        output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})

        return output

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        if self.sequential_computation:
            # warning: here only one `output` will be returned
            output = self._sequential_evaluation(model, x, **kwargs)
        else:
            x_target = self._expand_sample(x)
            output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)

        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz)

        # loss
        L_k = iw_data.get('L_k').mean(1)  # MC averaging
        loss = - L_k

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(iw_data.get('kl')),
                     'N_eff': iw_data.get('N_eff')},
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output


class ZScore(nn.Module):
    def __init__(self, momentum=0.001, eps=1e-05):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.initialized = False
        self.register_buffer("mean", torch.tensor(0.))
        self.register_buffer("variance", torch.tensor(1.))


    def update_statistics(self, x, momentum):
        self.mean = (1 - momentum) * self.mean + momentum * x.mean()
        self.variance = (1 - momentum) * self.variance + momentum * x.var()

    @torch.no_grad()
    def forward(self, x):
        z = (x - self.mean) / (self.eps + self.variance.sqrt())

        if self.training:

            if self.initialized:
                self.update_statistics(x, self.momentum)
            else:
                self.initialized = True
                self.update_statistics(x, 1)

        return z

    @torch.no_grad()
    def get_threshold(self, z_reject):
        return z_reject * (self.eps + self.variance.sqrt()) + self.mean


class Reinforce(VariationalInference):
    """
    Reinforce with optional baseline:

    * (a) in the general case:

    L = E_q [ log f(x, z) ], f(x,z) = p(x,z)/q(z|x)
    d/d theta L = d/d theta \sum_z q(z|x) log f(x,z)
             = \sum_z log f(x,z) d/d theta q(z|x) + \sum_z q(z_x) d/d theta f(x,z)
             = \sum_z q(z|x) log f(x,z) d/d theta log q(z|x) + \sum_z q(z,x) d/d theta f(x,z)
             = E_q [ log f(x,z) d/d theta log q(z|x) + d/d theta f(x,z)]


    * (b) when deriving from `phi`, the parameters of the posterior, in that case L_k will be only differentiated with regards to `theta`
    d/d_phi L_k =  E_q [ [ log( 1/K \sum_k w(x,z_k) ) - v_k ] d/d_phi log q(z|x), where v_k = w_m / \sum_m w_m

    in that case we define the score as E_q[ log( 1/K \sum_k w(x,z_k) ) - v_k ]

    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.control_variate_loss_weight = 0 if baseline is None else 1.
        assert self.sequential_computation == False

        # `score_from_phi` == True results in using case (b)
        self.score_from_phi = False

        # measure distribution of L1 for rejection sampling
        self.z_score_l1 = ZScore()

    def compute_control_variate(self, x: Tensor, **data: Dict[str, Tensor]) -> Tuple[Tensor, int]:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains `kwargs` and the outputs of the methods `compute_iw_bound` and `evaluate_model`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        if self.baseline is None:
            return torch.zeros((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype), 0

        baseline = self.baseline(x)
        n_nans = len(baseline[baseline!=baseline])

        return baseline.view((x.size(0), 1, 1, 1)), n_nans  # output of shape [bs, 1, 1, 1], Number of NaNs

    def compute_control_variate_l1(self, score, control_variate, weights=None):
        """L1 between the score function and its estimate"""

        if weights is None:
            weights = torch.ones_like(score)

        diff = (control_variate - score[:, :, :, None].detach())
        return (weights[..., None] * diff).abs().sum(3)  # sum over z

    def compute_reinforce_loss(self, score, control_variate, log_qz, weights=None):

        log_qz = log_qz.view(score.size(0), self.mc, self.iw, -1)

        if weights is None:
            weights = torch.ones_like(score)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        reinforce_loss = (score[:, :, :, None] - control_variate).detach() * log_qz

        # sum over iw: log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        return (weights[..., None] * reinforce_loss).sum((2, 3))  # sum over z and iw

    def normalized_importance_weights(self, log_f_xz):
        v = softmax(log_f_xz, dim=2)
        if v[v > 1 + _EPS].sum() > 0:
            print(f"~~~~ warning | in:normalized_importance_weights: v> 1 {v[v > 1 + _EPS]}")
        return v

    def compute_score(self, iw_data, mc_estimate):
        L_k, kl, log_f_xz = [iw_data[k] for k in ('L_k', 'kl', 'log_f_xz')]

        if self.score_from_phi:
            v = self.normalized_importance_weights(log_f_xz)
            score = L_k[:, :, None] - v
        else:
            score = L_k[:, :, None]

        if mc_estimate:
            score = score.mean(1, keepdim=True)

        return score

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, mc_estimate: bool = False, z_reject=0,
                **kwargs: Any) -> \
            Tuple[Tensor, Dict, Dict]:

        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=self.score_from_phi)
        L_k, kl, N_eff = [iw_data[k] for k in ('L_k', 'kl', 'N_eff')]

        # compute score l: dL(z)\dtheta = L dq(z) \ dtheta
        score = self.compute_score(iw_data, mc_estimate=mc_estimate)

        # compute control variate and MSE
        control_variate, _n_nans = self.compute_control_variate(x, mc_estimate=mc_estimate, **iw_data, **output,
                                                                **kwargs)
        control_variate_l1 = self.compute_control_variate_l1(score, control_variate)

        # rejection sampling according to the control variate L1
        if z_reject > 0:
            z_score_l1 = self.z_score_l1(control_variate_l1)
            reject_weights = (z_score_l1 < z_reject).float()

            reject_ratio = (1 - reject_weights).sum() / reject_weights.view(-1).shape[0]
            l1_threshold = self.z_score_l1.get_threshold(z_reject)

            if reject_ratio > 0.5 or (not self.z_score_l1.initialized):  # safety
                reject_weights = None
                reject_ratio = 0
        else:
            reject_weights = None
            reject_ratio = 0.
            l1_threshold = 0.

        # log filtered f_m to a file for debugging
        # if reject_weights is not None:
        #     with torch.no_grad():
        #         log_f_xz = iw_data.get('log_f_xz')
        #         for b, w_b in enumerate(reject_weights):
        #             for k, w_k in enumerate(w_b):
        #                 if w_k.sum() / len(w_k.view(-1)) < 1:
        #                     f_s = log_f_xz[b, k]
        #                     f_keep = f_s[w_k==1]
        #                     f_reject = f_s[w_k == 0]
        #                     p_f_keep = Normal(f_keep.mean(), f_keep.std())
        #                     print(f">> p(reject | keep) {p_f_keep.log_prob(f_reject).exp().mean().item():.2E}     ,p(keep | keep) {p_f_keep.log_prob(f_keep).exp().mean().item():.2E}")

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1)

        # reinforce loss
        reinforce_loss = self.compute_reinforce_loss(score, control_variate, log_qz, weights=reject_weights)

        # averaging MSE
        control_variate_l1_raw = control_variate_l1.mean((1, 2,))
        if reject_weights is None:
            control_variate_l1 = control_variate_l1_raw
        else:
            control_variate_l1 = self.compute_control_variate_l1(score, control_variate, weights=reject_weights)
            control_variate_l1 = control_variate_l1.sum(dim=(1, 2,)) / reject_weights.sum(dim=(1, 2,))

        # MC averaging
        reinforce_loss = reinforce_loss.mean(1)
        if reject_weights is not None:
            # the reinforce term (score - baseline) * dlogits is filtered using the rejection rule
            # however the L_k terms still contains all IW samples information
            # one intuitive solution is to weight each L_k term by the number of `active` samples.

            w_norm = 'exp'

            m_weights = reject_weights.sum(2)

            if w_norm == 'exp':
                w = m_weights / reject_weights.shape[2]
            else:
                w = torch.exp(m_weights - reject_weights.shape[2])

            w = w / w.sum(1, keepdim=True)
            _L_k = (w * L_k).sum(1)
            L_k = L_k.mean(1)
        else:
            _L_k = L_k = L_k.mean(1)

        # final loss
        loss = - _L_k - reinforce_loss + self.control_variate_loss_weight * control_variate_l1

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(kl),
                     'N_eff': N_eff,
                     'reinforce_loss': reinforce_loss,
                     'control_variate_l1': control_variate_l1,
                     'control_variate_l1_raw': control_variate_l1_raw,
                     'l1_threshold': l1_threshold,
                     'NaNs': _n_nans,
                     'rejected': reject_ratio}
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output


class Vimco(Reinforce):
    """
    VIMCO: https://arxiv.org/abs/1602.06725
    Inspired by https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_variate_loss_weight = 0  # the control variate doesn't have any parameter
        self.log_iw_m1 = np.log(self.iw - 1)

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, mc_estimate: bool = True, arithmetic=False, return_raw=False,
                                use_outer_samples=False, use_double: bool = True, **data: Dict[str, Tensor]) -> Tensor:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains the output of the method `compute_iw_bound`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        log_f_xz = data['log_f_xz']
        log_f_xz = log_f_xz.view(-1, self.mc, self.iw)
        _dtype = log_f_xz.dtype
        if use_double:
            log_f_xz = log_f_xz.double()

        if arithmetic:  # log \hat{f}(x, h^{-j}) using the arithmetic mean

            if use_outer_samples:

                mask = 1 - torch.eye(self.iw * self.mc, dtype=log_f_xz.dtype, device=log_f_xz.device)[None, :, :]
                _log_f_xz = log_f_xz[:, None, None, :, :].expand(-1, self.mc, self.iw, self.mc, self.iw)
                _log_f_xz = _log_f_xz.view(log_f_xz.size(0), self.mc * self.iw, self.mc * self.iw)

                # make sure to replace excluded samples with means so it doens't blow up to NAN with the exp (which gives `0` when multiplied by zero)
                _min, _ = _log_f_xz.min(dim=2, keepdim=True)
                _log_f_xz = (1 - mask) * _min + mask * _log_f_xz

                # compute the maximum for the log sum exp
                max, idx = _log_f_xz.max(dim=2, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_f_xz - max), dim=2)
                log_f_xz_hat = max.squeeze(2) + torch.log(
                    sum_exp) - self.log_mc_iw_m1  # adding eps should be necessary since the sum should be at least = exp(0)
                log_f_xz_hat = log_f_xz_hat.view(log_f_xz.size(0), self.mc, self.iw)

            else:
                mask = 1 - torch.eye(self.iw, dtype=log_f_xz.dtype, device=log_f_xz.device)[None, None, :, :]
                _log_f_xz = log_f_xz[:, :, None, :].expand(-1, self.mc, self.iw, self.iw)

                # make sure to replace excluded samples with means so it doens't blow up to NAN with the exp (which gives `0` when multiplied by zero)
                _min, _ = _log_f_xz.min(dim=3, keepdim=True)
                _log_f_xz = (1 - mask) * _min + mask * _log_f_xz

                # get the max for the LSE trick
                max, idx = _log_f_xz.max(dim=3, keepdim=True)

                sum_exp = torch.sum(mask * torch.exp(_log_f_xz - max), dim=3)
                log_f_xz_hat = max.squeeze(3) + torch.log(sum_exp) - self.log_iw_m1

        else:  # log \hat{f}(x, h^{-j}) using the geometric mean

            if use_outer_samples:
                log_f_xz_hat = (torch.sum(log_f_xz, dim=(1, 2), keepdim=True) - log_f_xz) / (self.mc * self.iw - 1)
            else:
                log_f_xz_hat = (torch.sum(log_f_xz, dim=2, keepdim=True) - log_f_xz) / (self.iw - 1)

        log_f_xz_samples = log_f_xz.unsqueeze(-1) + torch.diag_embed(log_f_xz_hat - log_f_xz)
        baseline = torch.logsumexp(log_f_xz_samples, dim=2) - self.log_iw

        # catchning nans

        _n_nans = len(baseline[baseline != baseline])
        if _n_nans > 0:

            print(">>> vimco:compute_control_variate: baseline NAN : ", len(baseline[baseline != baseline]))
            # print("scripts args = ", sys.argv)

            if torch.isnan(baseline).all():
                baseline[baseline != baseline] = 0
            else:
                baseline[baseline != baseline] = baseline[baseline == baseline].mean()

        if mc_estimate:
            baseline = baseline.mean(1, keepdim=True)

        if return_raw:
            return baseline.unsqueeze(-1), _n_nans
        else:
            return baseline.unsqueeze(-1).type(_dtype), _n_nans  # output of shape [bs, mc, iw, 1]


class ExactReinforce(Reinforce):
    """
    Compute the exact gradients. This only works for categorical prior and with N=1.
    """
    pass


class Lax(VariationalInference):
    """
    Relax: https://arxiv.org/abs/1711.00123
    """

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        raise NotImplementedError


def log(x):
    return torch.log(_EPS + x)


class Relax(VariationalInference):
    """
    Rebar: https://arxiv.org/abs/1703.07370
    Relax: https://arxiv.org/abs/1711.00123

    NB: in the RELAX paper, f(x, z) = log p(x,z) - log q(z|x)

    NB: this implementation is probably only 90% correct.

    todo: learn tau (setting tau as a parameter causes malloc error)
    """

    def __init__(self, *args, N=8, K=8, hdim=32, **kwargs):
        super().__init__(*args, **kwargs)

        if self.iw > 1:
            raise NotImplementedError

        # control variate model
        self.r_rho = Baseline(xdim=(N, K), nlayers=1, hdim=hdim)

        self.register_buffer("log_tau", torch.log(0.5 * torch.ones((1, N, 1))))
        # self.log_tau = nn.Parameter(torch.log(0.5 * torch.ones((1, N, 1)))) # todo: learning the temperature causes memory allocation error

    @property
    def tau(self):
        return self.log_tau.exp()

    def sigma(self, z, tau):
        return softmax(z / tau, dim=-1)

    def sample_posterior(self, logits):
        # sample noise
        u = torch.empty_like(logits).uniform_()
        v = torch.empty_like(logits).uniform_()

        # sample z ~ p(z | \theta)
        z = logits - log(-log(u))

        # b = H(z) <-> b ~ p(b | \theta) (Gumbel-Max trick)
        b_index = z.max(2, keepdim=True)[1]
        b = torch.zeros_like(z).scatter_(2, b_index, 1.0)
        b = b.detach()

        # sample z_tilde ~ p(z | b, \theta) (Appendix B)
        theta = softmax(logits, dim=-1)
        v_b = v.gather(dim=-1, index=b_index)
        z_i_eq_b = - log(-log(v))
        z_i_diff_b = - log(- log(v) / theta - log(v_b))
        z_tilde = torch.where(b == 1, z_i_eq_b, z_i_diff_b)

        return z, b, z_tilde

    def f(self, model, qz, pz, x, z):
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = batch_reduce(log_qz - log_pz)
        elbo = log_px_z - kl
        return elbo, kl, -log_px_z, px

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:

        assert isinstance(model, VAE), "only implemented for basic VAE"

        # expand sample
        bs, *dims = x.size()
        x = x[:, None].repeat(1, self.mc, *(1 for __ in dims)).contiguous().view(-1, *dims)

        qlogits = model.infer(x)
        qlogits.retain_grad()

        # sample b, z, z_tilde
        z, b, z_tilde = self.sample_posterior(qlogits)

        # p(z) and q(z|x)
        pz = PseudoCategorical(model.prior)
        qz = PseudoCategorical(qlogits)

        # compute f(x, b)
        f_b, kl, nll, px = self.f(model, qz, pz, x, b)

        # compute control variates
        sig_z = self.sigma(z, self.tau)
        f_z, *_ = self.f(model, qz, pz, x, sig_z)
        c_z = f_z + self.r_rho(sig_z)

        sig_z_tilde = self.sigma(z_tilde, self.tau)
        f_z_tilde, *_ = self.f(model, qz, pz, x, sig_z_tilde)
        c_z_tilde = f_z_tilde + self.r_rho(sig_z_tilde)

        # for debugging
        control_variate_mse = (c_z_tilde - f_b).pow(2).detach()

        # loss
        reinforce_loss = torch.sum((f_b - c_z_tilde).unsqueeze(1).detach() * qz.log_prob(b), 1)
        loss = - (
                f_b + reinforce_loss + c_z - c_z_tilde)  # todo: check if f_b terms is correct (doesn't work without that)

        def _reduce(x):
            _, *_dims = x.shape
            x = x.view(-1, self.iw, *_dims)
            return x.mean(1)

        # MC averaging
        loss, f_b, nll, kl, reinforce_loss, control_variate_mse = map(_reduce, (
            loss, f_b, nll, kl, reinforce_loss, control_variate_mse))

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': f_b,
                     'nll': nll,
                     'kl': kl,
                     'reinforce_loss': reinforce_loss,
                     'control_variate_mse': control_variate_mse}
        })

        if backward:

            # compute grads for theta
            loss.mean().backward(create_graph=True, retain_graph=True)

            # compute the variance of the gradients with regards to the logits
            grads_var = (qlogits.grad.mean(0) ** 2).mean()

            # set grads of tau and rho to zero (they were modified by the backward pass)
            self.zero_grad()

            # looping over rho params: super slow..
            for k, v in list(self.named_parameters())[::-1]:

                # compute d grads_var / dv
                grads_v, = torch.autograd.grad(
                    [grads_var], [v], retain_graph=True, allow_unused=True)

                # assign gradients manually
                if grads_v is not None:
                    v.grad = grads_v.data

        return loss, diagnostics, {'x_': px, 'z': b, 'qz': qz, 'pz': pz, 'qlogits': qlogits}
