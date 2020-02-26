import warnings
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

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        # expand input to get MC and IW samples: [mc, iw, bs, ...]
        bs, *dims = x.size()
        __x = self._expand_sample(x)

        # forward pass
        output = model(__x, **kwargs)
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # compute log probs
        log_px_z = reduce(px.log_prob(__x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute log f(x, z) (ELBO)
        log_f_xz = log_px_z + log_pz - log_qz

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(bs, self.mc, self.iw)

        # IW averaging: L_k = E_q(z_{1..K}|x)[ log 1/K \sum_i f(x, z_i)]
        L_k = log_sum_exp(log_f_xz, dim=2) - self.log_iw

        # MC averaging
        L_k = torch.mean(L_k, dim=1)

        # gradient ascent
        loss = - L_k

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            w = (log_pz - log_qz).exp().view(bs, self.mc, self.iw)
            N_eff = torch.sum(w, 2) ** 2 / torch.sum(w ** 2, 2)
            N_eff = N_eff.mean(1)  # MC
        else:
            N_eff = torch.ones_like(loss)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(log_qz - log_pz),
                     'N_eff': N_eff},
        })

        return loss, diagnostics, output


class Reinforce(Estimator):

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        # expand input to get MC and IW samples: [mc, iw, bs, ...]
        bs, *dims = x.size()
        __x = self._expand_sample(x)

        # forward pass
        output = model(__x, **kwargs)
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # compute log probs
        log_px_z = reduce(px.log_prob(__x))
        log_pz = reduce(pz.log_prob(z))
        __log_qz = qz.log_prob(z)
        log_qz = reduce(__log_qz)

        # compute f(x, z) (ELBO)
        f_xz = log_px_z + log_pz - log_qz

        # view f as shape [bs, mc, iw]
        f_xz = f_xz.view(bs, self.mc, self.iw)

        # IW averaging: L_k = E[log 1/k \sum_i f(x, z_i)]
        L_k = log_sum_exp(f_xz, dim=2) - self.log_iw

        # baseline: b(x) + c
        if self.baseline is not None:
            baseline = self.baseline(x)
            # expand to match format [mc, bs]
            baseline = baseline[None].repeat(1, self.mc, *(1 for _ in dims))
            # MSE between baseline and f(x,z)
            baseline_loss = (baseline - L_k.detach()).pow(2)
        else:
            baseline = torch.zeros_like(L_k)
            baseline_loss = torch.zeros_like(L_k)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        # log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        __log_qz = __log_qz.view(bs, self.mc, self.iw, -1).sum(2)
        reinforce_loss = (L_k - baseline)[:, :, None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum(-1)  # sum over z

        # MC averaging
        L_k, reinforce_loss, baseline_loss = map(lambda x: x.mean(1), (L_k, reinforce_loss, baseline_loss))

        # final loss
        loss = - L_k - reinforce_loss + baseline_loss

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            w = (log_pz - log_qz).exp().view(bs, self.mc, self.iw)
            N_eff = torch.sum(w, 2) ** 2 / torch.sum(w ** 2, 2)
            N_eff = N_eff.mean(1)  # MC
        else:
            N_eff = torch.ones_like(loss)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(log_qz - log_pz),
                     'N_eff': N_eff,
                     'reinforce': reinforce_loss,
                     'baseline': baseline_loss}
        })

        return loss, diagnostics, output


class Vimco(VariationalInference):
    """
    Inspired by https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py
    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline

    def forward(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        # expand input to get MC and IW samples: [mc, iw, bs, ...]
        bs, *dims = x.size()
        __x = self._expand_sample(x)

        # forward pass
        output = model(__x, **kwargs)
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # compute log probs
        log_px_z = reduce(px.log_prob(__x))
        log_pz = reduce(pz.log_prob(z))
        __log_qz = qz.log_prob(z)
        log_qz = reduce(__log_qz)

        # compute log f(x, z) = log p(x, z) / q(z|x)
        log_f_xz = log_px_z + log_pz - log_qz

        # view f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(bs, self.mc, self.iw)

        # IW averaging: L_k = E[log 1/k \sum_i f(x, z_i)]
        L_k = torch.logsumexp(log_f_xz, dim=2) - self.log_iw

        # log \hat{f}(x, h^{-j}) using the geometric mean
        log_f_xz_hat = (torch.sum(log_f_xz, dim=2, keepdim=True) - log_f_xz) / (self.iw - 1)
        log_f_xz_samples = log_f_xz.unsqueeze(-1) + torch.diag_embed(log_f_xz_hat - log_f_xz)
        baseline = torch.logsumexp(log_f_xz_samples, dim=2) - self.log_iw

        # for debugging
        baseline_loss = (baseline - L_k.unsqueeze(2)).pow(2).mean(2).detach()

        # \sum L_(h_j| -j) * log q(z_j | x)
        __log_qz = __log_qz.view(bs, self.mc, self.iw, -1)
        reinforce_loss = (L_k.unsqueeze(2) - baseline).unsqueeze(-1).detach() * __log_qz
        reinforce_loss = reinforce_loss.sum((2, 3))  # sum over iw samples and z

        # MC averaging
        L_k, reinforce_loss, baseline_loss = map(lambda x: x.mean(1), (L_k, reinforce_loss, baseline_loss))

        # final loss
        loss = - L_k - reinforce_loss

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            w = (log_pz - log_qz).exp().view(bs, self.mc, self.iw)
            N_eff = torch.sum(w, 2) ** 2 / torch.sum(w ** 2, 2)
            N_eff = N_eff.mean(1)  # MC
        else:
            N_eff = torch.ones_like(loss)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(log_qz - log_pz),
                     'N_eff': N_eff,
                     'reinforce': reinforce_loss,
                     'baseline': baseline_loss}
        })

        return loss, diagnostics, output