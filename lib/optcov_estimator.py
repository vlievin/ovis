from booster import Diagnostic
from torch import nn, Tensor

from .estimators import *
from .model import PseudoCategorical
from .utils import *

_EPS = 1e-18


class OptCovReinforce(VariationalInference):
    """
    Overleaf: https://www.overleaf.com/project/5d84f62f0c4fb30001e934ac

    c_\mathrm{opt}(x) = \frac{\sum_k \sum_m h^T_{mk} \sum_{m'} h_{m'k} f_{m'k} }{\sum_k  \sum_m h^T_{mk} \sum_{m'} h_{m'k} }

    """

    def __init__(self, *args, baseline: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline is not None  # small hack to allow testing the estimator without the control variate.

    def f(self, model, qz, pz, x, z):
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = batch_reduce(log_qz - log_pz)
        elbo = log_px_z - kl
        return elbo, kl, -log_px_z, px

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, exclude_sample=True, scalar_baseline=False, **kwargs: Any) -> Tuple[
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
        log_px_z = batch_reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)
        kl = batch_reduce(log_qz - log_pz)
        nll = - log_px_z

        # warning: detach kl such as L_k is only differentiable with regard to theta
        kl = kl.detach()

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(bs, self.mc, self.iw)

        # IW-ELBO: L_k = E_q(z_{1..K} | x) [ log 1/K \sum_{i=1..K} f(x, z_i)]
        L_k = torch.logsumexp(log_f_xz, dim=2) - self.log_iw

        # N_eff
        N_eff = self.effective_sample_size(log_pz, log_qz)

        # d q(z|x) / d qlogits
        d_qlogits = torch.autograd.grad(
            [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)[0]

        # reshaping d_qlogits and qlogits
        d_qlogits = d_qlogits.view(bs, self.mc, self.iw, *d_qlogits.size()[1:])
        qlogits = qlogits.view(bs, self.mc, self.iw, *qlogits.size()[1:])
        nll = nll.view(bs, self.mc, self.iw).mean(2)

        # vector control variate
        with torch.no_grad():
            v_mk = torch.exp(log_f_xz - torch.logsumexp(log_f_xz, dim=2, keepdim=True))

            # using notation from the overleaf doc
            h = d_qlogits.view(bs, self.mc, self.iw, -1)
            f_mk = (L_k[:,:,None] - v_mk)

            # compute Amm and bm
            Amn = torch.einsum("bkmh, bknh -> bkmn", [h, h])
            bm = torch.einsum("bkmh, bkh -> bkm", [h, torch.sum(h * f_mk[:,:,:,None], dim=2)])

            # MC averaging
            if exclude_sample:
                Amn = (Amn.sum(1, keepdim=True) - Amn) / (self.mc-1)
                bm = (bm.sum(1, keepdim=True) - bm) / (self.mc-1)
            else:
                Amn = Amn.mean(1, keepdim=True)
                bm = bm.mean(1, keepdim=True)


            # scalar baseline
            if scalar_baseline:
                Amn = Amn.sum((-2, -1,), keepdim=True)
                bm = bm.sum((-1,), keepdim=True)

            # compute Amn^{-1}
            shp = Amn.shape
            Amn = Amn.view(-1, *shp[-2:])
            Amn_inv = torch.pinverse(Amn, rcond=1e-18)  # use pseudo inverse for stability
            Amn_inv = Amn_inv.view(shp)

            # baseline: C_opt = Amn^{-1} bm
            c_opt = (Amn_inv @ bm.unsqueeze(-1)).squeeze(-1)

            # reinforce score
            score = (f_mk[:, :, :] - c_opt) #if self.baseline else f_mk

            control_variate_mse = (f_mk[:, :, :] - c_opt).pow(2).mean((1, 2))

        # reinforce loss
        __log_qz = log_qz.view(bs, self.mc, self.iw, -1)
        reinforce_loss = score[:, :, :, None].detach() * __log_qz
        reinforce_loss = reinforce_loss.sum((2, 3))  # sum over z and iw

        # MC averaging
        nll, L_k, reinforce_loss = map(lambda x: x.mean(1), (nll, L_k, reinforce_loss))

        # nll gives the gradients for theta
        loss = -L_k - reinforce_loss

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
