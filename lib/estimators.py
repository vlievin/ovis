from typing import *

import numpy as np
from booster import Diagnostic
from torch import nn, Tensor

from .utils import *


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

    def forward(self, model, x, **kwargs):
        raise NotImplementedError


class VariationalInference(Estimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_elbo(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple:
        """
        Compute the Importance-Weighted ELBO.

        L_k = E_{q(z_{1..K} | x)} [ log 1/K \sum_{i=1..K} f(x, z_i)], f(x, z) = p(x,z) / q(z|x)

        :param model:
        :param x:
        :param kwargs:
        :return:
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

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

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

        return loss, diagnostics, output


class Reinforce(VariationalInference):
    """
    Reinforce with optional baseline:

    \nabla L_k = (L_k - b(x)) \nabla q(z | x)

    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

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

        return loss, diagnostics, output


class Vimco(VariationalInference):
    """
    VIMCO: https://arxiv.org/abs/1602.06725
    Inspired by https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py
    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        bs, *dims = x.size()
        output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl = self.compute_elbo(model, x, **kwargs)

        # log \hat{f}(x, h^{-j}) using the geometric mean
        log_f_xz_hat = (torch.sum(log_f_xz, dim=2, keepdim=True) - log_f_xz) / (self.iw - 1)
        log_f_xz_samples = log_f_xz.unsqueeze(-1) + torch.diag_embed(log_f_xz_hat - log_f_xz)
        baseline = torch.logsumexp(log_f_xz_samples, dim=2) - self.log_iw

        # for debugging
        control_variate_mse = (baseline - L_k.unsqueeze(2)).pow(2).mean(2).detach()

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

        return loss, diagnostics, output
