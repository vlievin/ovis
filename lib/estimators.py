import numpy as np
from booster import Diagnostic
from torch import nn, Tensor
from torch.nn.functional import softmax

from .baseline import Baseline
from .model import PseudoCategorical, VAE
from .utils import *

_EPS = 1e-18


class Estimator(nn.Module):
    def __init__(self, beta: float = 1, mc: int = 1, iw: int = 1, freebits=0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mc = mc
        self.iw = iw
        self.log_iw = np.log(iw)
        self.freebits = FreeBits(freebits)
        self.detach_qlogits = False

    def _expand_sample(self, x):
        bs, *dims = x.size()
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


class VariationalInference(Estimator):
    """
    Variational Inference. Using this estimator requires the model to be compatible with the reparametrization trick.

    """

    def compute_iw_bound(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor]) -> Dict[str, Tensor]:
        """
        Compute the importance weighted bound:

         L_k = E_{q(z_{1..K} | x)} [ log 1/K \sum_{i=1..K} f(x, z_i)], f(x, z) = p(x,z) / q(z|x)

         In this expression the KLs are concatenated by stochastic layer so the freebits can be applied to each of them.

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
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
        if self.detach_qlogits:
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

    def evaluate_model(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Dict[str, Tensor]:

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

        output = self.evaluate_model(model, x, **kwargs)
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


class Reinforce(VariationalInference):
    """
    Reinforce with optional baseline:

    L = E_q [ log f(x, z) ], f(x,z) = p(x,z)/q(z|x)
    \nabla L = \nabla \sum_z q(z|x) log f(x,z)
             = \sum_z log f(x,z) \nabla q(z|x) + \sum_z q(z_x) \nabla f(x,z)
             = \sum_z q(z|x) log f(x,z) \nabla log q(z|x) + \sum_z q(z,x) \nabla f(x,z)
             = E_q [ log f(x,z) \nabla log q(z|x) + \nabla f(x,z)]

    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.control_variate_loss_weight = 0 if baseline is None else 1.

    def compute_control_variate(self, x: Tensor, **data: Dict[str, Tensor]) -> Tensor:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains `kwargs` and the outputs of the methods `compute_iw_bound` and `evaluate_model`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        if self.baseline is None:
            return torch.zeros((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype)

        baseline = self.baseline(x)
        return baseline.view((x.size(0), 1, 1, 1))  # output of shape [bs, 1, 1, 1]

    def compute_control_variate_mse(self, L_k, control_variate):
        """mse between the score function and its estimate"""
        return (control_variate - L_k[:, :, None, None].detach()).pow(2).sum(3).mean(2)  # <-> sum(z).mean(iw)

    def compute_reinforce_loss(self, L_k, control_variate, log_qz):
        log_qz = log_qz.view(L_k.size(0), self.mc, self.iw, -1)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        reinforce_loss = (L_k[:, :, None, None] - control_variate).detach() * log_qz
        # sum over iw: log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        return reinforce_loss.sum((2, 3))  # sum over z and iw

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz)
        L_k, kl, log_f_xz, N_eff = [iw_data[k] for k in ('L_k', 'kl', 'log_f_xz', 'N_eff')]

        # baseline: b(x) + c
        control_variate = self.compute_control_variate(x, **iw_data, **output, **kwargs)
        control_variate_mse = self.compute_control_variate_mse(L_k, control_variate)

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1)

        # reinforce loss
        reinforce_loss = self.compute_reinforce_loss(L_k, control_variate, log_qz)

        # MC averaging
        L_k, reinforce_loss, control_variate_mse = map(lambda x: x.mean(1), (L_k, reinforce_loss, control_variate_mse))

        # final loss
        loss = - L_k - reinforce_loss + self.control_variate_loss_weight * control_variate_mse

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(kl),
                     'N_eff': N_eff,
                     'reinforce_loss': reinforce_loss,
                     'control_variate_mse': control_variate_mse}
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

    @torch.no_grad()
    def compute_control_variate(self, x: Tensor, mc_estimate: bool = True, **data: Dict[str, Tensor]) -> Tensor:
        """Compute the baseline that will be substracted to the score L_k,
        `data` contains the output of the method `compute_iw_bound`.
        The output shape should be of size 4 and matching the shape [bs, mc, iw, nz]"""

        log_f_xz = data['log_f_xz']
        log_f_xz = log_f_xz.view(-1, self.mc, self.iw)

        # log \hat{f}(x, h^{-j}) using the geometric mean
        log_f_xz_hat = (torch.sum(log_f_xz, dim=2, keepdim=True) - log_f_xz) / (self.iw - 1)
        log_f_xz_samples = log_f_xz.unsqueeze(-1) + torch.diag_embed(log_f_xz_hat - log_f_xz)
        baseline = torch.logsumexp(log_f_xz_samples, dim=2) - self.log_iw

        if mc_estimate:
            baseline = baseline.mean(1, keepdim=True)

        return baseline.unsqueeze(-1)  # output of shape [bs, mc, iw, 1]


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
