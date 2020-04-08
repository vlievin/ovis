"""
work in progress
"""


from booster import Diagnostic
from torch import nn, Tensor

from lib.estimators import *
from lib.utils import *

_EPS = 1e-18


class StructuredReinforce(Reinforce):
    """
    Let be a LVAE model with L layers with
        * p(x, z) = p(x | z)p(z)
        * p(z) = p(z_L) \prod_{i=1}^{L-1} p(z_i | z_{>i})
        * q(z | x) = p(z_L | x) \prod_{i=1}^{L-1} p(z_i | z_{>i}, x)

    Let denote theta_i the parameters of the variational distribution at the layer i and the set theta = {theta}_{i=1..L}.

    The evidence lower bound is given by L = E_q(z|x) [ log f(x,z) ] where f(x,z) = p(x,z) / q(z|x).

    In the classic REINFORCE, the derivation of the gradients yields
    \nabla_theta L = E_q(z|x) [ log f(x,z) \nabla_theta q(z|x)] + E_q(z|x) [ \nabla log f(x, z) ] .

    However, using the dependencie structure of the LVAE, one can notice that:

    \nabla_theta_i L = E_q( z_{>i} | x) [ \nabla\theta_i E_q( z_\leq{i} | x, z_{>i}) [ log f(x, z_\leq{i} | z_{>i}) ] + 0 ]

    """


    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        bs, *dims = x.size()
        # output, L_k, N_eff, log_f_xz, log_px_z, log_pz, log_qz, kl = self.compute_elbo(model, x, **kwargs)

        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz, baseline = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz', 'baseline')]


        # compute reinforce loss for each layer stochastic layer
        for i, log_qz_i in enumerate(log_qz):

            iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz)
            L_k, kl, log_f_xz, N_eff = [iw_data[k] for k in ('L_k', 'kl', 'log_f_xz', 'N_eff')]

            # baseline: b(x) + c
            control_variate = self.compute_control_variate(x, **iw_data)
            control_variate_mse = self.compute_control_variate_l1(L_k, control_variate)

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




class StructuredVariationalInference___(VariationalInference):

    def compute_partial_elbo_(self, log_px_z, log_pzs, log_qzs):
        # todo: move in the main class

        # E_q[ log p(z) - log q(z) ]
        log_pz = torch.cat(log_pzs, 1)
        log_qz = torch.cat(log_qzs, 1)
        kl = batch_reduce(self.freebits(log_qz - log_pz))  # todo: check freebits behaviour

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # view log f as shape [bs, mc, iw]
        log_f_xz = log_f_xz.view(-1, self.mc, self.iw)

        # IW-ELBO: L_k = E_q(z_{1..K} | x) [ log 1/K \sum_{i=1..K} f(x, z_i)]
        L_k = torch.logsumexp(log_f_xz, dim=2) - self.log_iw

        return {'L_k': L_k, 'kl': kl}

    def compute_partial_elbos(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Iterable[Dict]:
        bs, *dims = x.size()
        __x = self._expand_sample(x)

        # forward pass
        output = model(__x, **kwargs)
        px, z, qz, pz, baseline = [output[k] for k in ['px', 'z', 'qz', 'pz', 'baseline']]

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(__x))
        log_pzs = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
        log_qzs = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

        # for layer i = 1..L
        for i, log_qz in enumerate(log_qzs):
            log_pz_i = log_pzs[:i + 1]
            log_qz_i = log_qzs[:i + 1]
            data = self.compute_partial_elbo_(log_px_z, log_pz_i, log_qz_i)
            N_eff_i = self.effective_sample_size(log_pz_i, log_qz_i)
            log_qz = log_qz.view(bs, self.mc, self.iw, -1)
            baseline_i = baseline[i]
            if baseline_i is None:
                baseline_i = torch.zeros_like(data['L_k'])
            data.update({'index': i, 'log_qz': log_qz, 'N_eff': N_eff_i, 'log_px_z': log_px_z,
                         'baseline': baseline_i.view(bs, self.mc, self.iw)})
            yield data


class StructuredReinforce(StructuredVariationalInference___):
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

    def reinforce_loss(self, score: Tensor, log_qz: Tensor, control_variate: Optional[Tensor] = None):
        if control_variate is not None:
            score = score[:, :, None] - control_variate

        # reinforce loss : \nabla L = (f(x,z) - b(x)) \nabla loq q(z|x)
        reinforce_loss = score.unsqueeze(-1).detach() * log_qz
        # sum over the latent dimensions and iw samples
        reinforce_loss = reinforce_loss.sum((2, 3))
        return reinforce_loss

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        bs, *dims = x.size()
        partial_elbos = list(self.compute_partial_elbos(model, x, **kwargs))

        # baseline: b(x) + c # todo: one baseline per stage (b(x) + b(z>i))
        if self.baseline is not None:
            control_variate = self.baseline(x)
            # expand to match format [mc, bs]
            control_variate = control_variate[:, None, None].repeat(1, self.mc, self.iw)
        else:
            control_variate = None

        # compute reinforce loss
        # NB: the last L_k and kl values correspond to the proper value (full model)
        reinforce_loss = 0
        control_variate_mse = 0
        for d in partial_elbos:
            L_k, kl, log_qz, N_eff, log_px_z, baseline = [d[k] for k in
                                                          ['L_k', 'kl', 'log_qz', 'N_eff', 'log_px_z', 'baseline']]

            if control_variate is not None:
                baseline = baseline + control_variate

            l = self.reinforce_loss(L_k, log_qz, control_variate=baseline)
            reinforce_loss = reinforce_loss + l

            # baseline loss
            control_variate_mse = control_variate_mse + (baseline - L_k[:, :, None].detach()).pow(2).mean(2)

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

        return loss, diagnostics, {}
