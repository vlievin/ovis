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

    def _expand_sample(self, x):
        bs, *dims = x.size()
        x = x[None, None].repeat(self.mc, self.iw, *(1 for _ in x.size()))
        # flatten everything into the batch dimension
        return x.view(-1, *dims)

    def _reduce_sample(self, x):
        _, *dims = x.size()
        x = x.view(self.mc, self.iw, -1, *dims)
        return x.mean((0, 1,))

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

        # compute f(x, z) (ELBO)
        f_xz = log_px_z + log_pz - log_qz

        # view f as shape [mc, iw, bs]
        f_xz = f_xz.view(self.mc, self.iw, bs)

        # IW averaging: L_k = E[log 1/k \sum_i f(x, z_i)]
        L_k = log_sum_exp(f_xz, dim=1) - np.log(f_xz.shape[1])

        # MC averaging
        L_k = torch.mean(L_k, dim=0)

        # gradient ascent
        loss = - L_k

        # compute effective sample size
        w = (log_pz - log_qz).exp().view(self.mc, self.iw, bs)
        N_eff = torch.sum(w, 1) ** 2 / torch.sum(w ** 2, 1)
        N_eff = N_eff.mean(0)

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

        # view f as shape [mc, iw, bs]
        f_xz = f_xz.view(self.mc, self.iw, bs)

        # IW averaging: L_k = E[log 1/k \sum_i f(x, z_i)]
        L_k = log_sum_exp(f_xz, dim=1) - np.log(f_xz.shape[1])

        # L_k
        score = - L_k

        # baseline: b(x) + c
        if self.baseline is not None:
            baseline = self.baseline(x)
            # expand to match format [mc, bs]
            baseline = baseline[None].repeat(self.mc, *(1 for _ in baseline.size()))
            # MSE between baseline and f(x,z)
            baseline_loss = (baseline - score.detach()).pow(2)
        else:
            baseline = torch.zeros_like(L_k)
            baseline_loss = torch.zeros_like(L_k)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        # log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        __log_qz = __log_qz.view(self.mc, self.iw, bs, -1).sum(1)
        reinforce_loss = (score - baseline)[:, :, None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum(-1)  # sum over z

        # MC averaging
        L_k, reinforce_loss, baseline_loss = map(lambda x: x.mean(0), (L_k, reinforce_loss, baseline_loss))

        # final loss, notice the use of `log_px_z` to enable gradients in the generative model
        loss = - L_k + reinforce_loss + baseline_loss

        # compute effective sample size
        w = (log_pz - log_qz).exp().view(self.mc, self.iw, bs)
        N_eff = torch.sum(w, 1) ** 2 / torch.sum(w ** 2, 1)
        N_eff = N_eff.mean(0)

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

        # view f as shape [mc, iw, bs]
        f_xz = f_xz.view(self.mc, self.iw, bs)

        # IW averaging: L_k = E[log 1/k \sum_i f(x, z_i)]
        L_k = log_sum_exp(f_xz, dim=1) - np.log(f_xz.shape[1])

        # L_k
        score = - L_k

        # baseline: b(x) + c
        if self.baseline is not None:
            baseline = self.baseline(x)
            # expand to match format [mc, bs]
            baseline = baseline[None].repeat(self.mc, *(1 for _ in baseline.size()))
            # MSE between baseline and f(x,z)
            baseline_loss = (baseline - score.detach()).pow(2)
        else:
            baseline = torch.zeros_like(L_k)
            baseline_loss = torch.zeros_like(L_k)

        # \nabla loss = [ f(x, z) - b(x) ] \nabla log q_theta(z | x)
        # log Q(z_{1..K} | x) = \sum_{i=1..K} log q(z_i | x)
        __log_qz = __log_qz.view(self.mc, self.iw, bs, -1).sum(1)
        reinforce_loss = (score - baseline)[:, :, None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum(-1)  # sum over z

        # MC averaging
        L_k, reinforce_loss, baseline_loss = map(lambda x: x.mean(0), (L_k, reinforce_loss, baseline_loss))

        # final loss, notice the use of `log_px_z` to enable gradients in the generative model
        loss = - L_k + reinforce_loss + baseline_loss

        # compute effective sample size
        if self.iw > 1:
            w = (log_pz - log_qz).exp()
            N_eff = torch.sum(w, 1) ** 2 / torch.sum(w ** 2, 1)
            N_eff = N_eff.mean(0)
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