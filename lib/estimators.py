from typing import *

import numpy as np
from booster import Diagnostic
from torch import nn, Tensor
from torch.nn.functional import softmax

from .baseline import Baseline
from .model import PseudoCategorical
from .utils import *

_EPS = 1e-18


class Estimator(nn.Module):
    def __init__(self, beta: float = 1, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__()
        self.beta = beta
        self.mc = mc
        self.iw = iw
        self.log_iw = np.log(iw)

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

    def compute_elbo(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple:
        """
        Compute the Importance-Weighted ELBO.

        L_k = E_{q(z_{1..K} | x)} [ log 1/K \sum_{i=1..K} f(x, z_i)], f(x, z) = p(x,z) / q(z|x)

        :param model: VAE
        :param x: data
        :param kwargs: arguments passed to the model.forward method
        :return: model's output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl
        """

        # expand input to get MC and IW samples: [mc, iw, bs, ...]
        bs, *dims = x.size()
        __x = self._expand_sample(x)

        # forward pass
        output = model(__x, **kwargs)
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = reduce(px.log_prob(__x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = reduce(log_qz - log_pz)

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(bs, self.mc, self.iw)

        # IW-ELBO: L_k = E_q(z_{1..K} | x) [ log 1/K \sum_{i=1..K} f(x, z_i)]
        L_k = log_sum_exp(log_f_xz, dim=2) - self.log_iw

        # N_eff
        N_eff = self.effective_sample_size(log_pz, log_qz)

        return output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl

    def effective_sample_size(self, log_pz, log_qz):
        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2
        :param log_pz: log p(z) of shape [bs * mc * iw, ...]
        :param log_qz: loq q(z) of shape [bs * mc * iw, ...]
        :return: effective_sample_size
        """

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            w = reduce(log_pz - log_qz).exp().view(-1, self.mc, self.iw)
            N_eff = torch.sum(w, 2) ** 2 / torch.sum(w ** 2, 2)
            N_eff = N_eff.mean(1)  # MC
        else:
            x = (log_pz).view(-1, self.mc, self.iw)
            N_eff = torch.ones_like(x[:, 0, 0])

        return N_eff

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl = self.compute_elbo(model, x, **kwargs)

        # loss
        L_k = L_k.mean(1)  # MC averaging
        loss = - L_k

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(kl),
                     'N_eff': N_eff},
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

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        bs, *dims = x.size()
        output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl = self.compute_elbo(model, x, **kwargs)

        # baseline: b(x) + c
        if self.baseline is not None:
            control_variate = self.baseline(x)
            # expand to match format [mc, bs]
            control_variate = control_variate[:, None].repeat(1, self.mc)
            # MSE between baseline and f(x,z)
            control_variate_mse = (control_variate - L_k.detach()).pow(2)
        else:
            control_variate = torch.zeros_like(L_k)
            control_variate_mse = torch.zeros_like(L_k)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        # log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        __log_qz = log_qz.view(bs, self.mc, self.iw, -1).sum(2)
        reinforce_loss = (L_k - control_variate)[:, :, None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum(-1)  # sum over z

        # MC averaging
        L_k, reinforce_loss, control_variate_mse = map(lambda x: x.mean(1), (L_k, reinforce_loss, control_variate_mse))

        # final loss
        loss = - L_k - reinforce_loss + control_variate_mse

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


class Vimco(VariationalInference):
    """
    VIMCO: https://arxiv.org/abs/1602.06725
    Inspired by https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py
    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        bs, *dims = x.size()
        output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl = self.compute_elbo(model, x, **kwargs)

        with torch.no_grad():
            # log \hat{f}(x, h^{-j}) using the geometric mean
            log_f_xz_hat = (torch.sum(log_f_xz, dim=2, keepdim=True) - log_f_xz) / (self.iw - 1)
            log_f_xz_samples = log_f_xz.unsqueeze(-1) + torch.diag_embed(log_f_xz_hat - log_f_xz)
            baseline = torch.logsumexp(log_f_xz_samples, dim=2) - self.log_iw

            # for debugging
            control_variate_mse = (baseline - L_k.unsqueeze(2)).pow(2).mean(2)

        # \sum L_(h_j| -j) * log q(z_j | x)
        __log_qz = log_qz.view(bs, self.mc, self.iw, -1)
        reinforce_loss = (L_k.unsqueeze(2) - baseline).unsqueeze(-1).detach() * __log_qz
        reinforce_loss = reinforce_loss.sum((2, 3))  # sum over iw samples and z

        # MC averaging
        L_k, reinforce_loss, control_variate_mse = map(lambda x: x.mean(1), (L_k, reinforce_loss, control_variate_mse))

        # final loss
        loss = - L_k - reinforce_loss

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
    """

    def __init__(self, *args, N=8, K=8, hdim=32, **kwargs):
        super().__init__(*args, **kwargs)

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
        log_px_z = reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = reduce(log_qz - log_pz)
        elbo = log_px_z - kl
        return elbo, kl, -log_px_z, px

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
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

            # looping over rho params: super slow..
            for k, v in list(self.r_rho.named_parameters())[::-1]:

                # compute d grads_var / dv
                grads_v = torch.autograd.grad(
                    [grads_var], [v], grad_outputs=torch.ones_like(grads_var), retain_graph=True, allow_unused=True)[0]

                # assign gradients manually
                if grads_v is not None:
                    v.grad = grads_v.data

        return loss, diagnostics, {'x_': px, 'z': b, 'qz': qz, 'pz': pz, 'qlogits': qlogits}



class OptCovReinforce(VariationalInference):
    """
    Overleaf: https://www.overleaf.com/project/5d84f62f0c4fb30001e934ac

    c_\mathrm{opt}(x) = \frac{\sum_k \sum_m h^T_{mk} \sum_{m'} h_{m'k} f_{m'k} }{\sum_k  \sum_m h^T_{mk} \sum_{m'} h_{m'k} }

    However here I am using :

    \nabla E_q[L_k] = E_q[ (f(x,z) - copt) h + \nabla f(x,z) ]

    where:
    * f(x,z) = log 1/k \sum_{i=1..K} p(x,z_i) / q(z_i | x)
    * h = \nabla q(z|x)

    """

    def __init__(self, *args, baseline: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline is not None # small hack to allow testing the estimator without the control variate.

    def f(self, model, qz, pz, x, z):
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = reduce(log_qz - log_pz)
        elbo = log_px_z - kl
        return elbo, kl, -log_px_z, px

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
        bs, *dims = x.shape

        # expand sample
        x = self._expand_sample(x)

        qlogits = model.infer(x)
        qlogits.retain_grad()

        # p(z) and q(z|x)
        pz = PseudoCategorical(model.prior)
        qz = PseudoCategorical(qlogits)

        # z ~ q(z|x)
        z = qz.sample()

        # p(x|z)
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = reduce(log_qz - log_pz)
        nll = - log_px_z

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(bs, self.mc, self.iw)

        # IW-ELBO: L_k = E_q(z_{1..K} | x) [ log 1/K \sum_{i=1..K} f(x, z_i)]
        L_k = log_sum_exp(log_f_xz, dim=2) - self.log_iw

        # N_eff
        N_eff = self.effective_sample_size(log_pz, log_qz)

        # d q(z|x) / d qlogits
        d_qlogits = torch.autograd.grad(
            [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)[0]

        # reshaping d_qlogits and qlogits
        d_qlogits = d_qlogits.view(bs, self.mc, self.iw, *d_qlogits.size()[1:])
        qlogits = qlogits.view(bs, self.mc, self.iw, *qlogits.size()[1:])
        nll = nll.view(bs, self.mc, self.iw).mean(2)

        # control variate
        with torch.no_grad():

            #using notation from the paper
            h = d_qlogits
            f_mk = L_k

            # compute h and h * f
            sum_h_m = h.sum(dim=2).view(bs, self.mc, -1)
            sum_hf_m = (h * f_mk[:, :, None, None, None]).sum(dim=2).view(bs, self.mc, -1)

            # compute c_opt
            num = torch.sum((sum_h_m * sum_hf_m).sum(-1), dim=1)
            den = torch.sum((sum_h_m * sum_h_m).sum(-1), dim=1)
            c_opt = num / den

            # reinforce score
            score = (f_mk - c_opt[:, None]) if self.baseline else f_mk

            control_variate_mse = (f_mk - c_opt[:, None]).pow(2).mean((1,))

        # reinforce loss
        __log_qz = log_qz.view(bs, self.mc, self.iw, -1).sum(2)
        reinforce_loss = score[:,:,None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum(-1)  # sum over z

        # MC averaging
        nll, L_k, reinforce_loss = map(lambda x: x.mean(1), (nll, L_k, reinforce_loss))

        # nll gives the gradients for theta
        loss = - L_k - reinforce_loss

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': nll,
                     'kl': kl,
                     'N_eff': N_eff,
                     'reinforce_loss': reinforce_loss,
                     'control_variate_mse': control_variate_mse}
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, {'x_': px, 'z': z, 'qz': qz, 'pz': pz, 'qlogits': qlogits}