from torch import softmax

from .base import _EPS
from .vi import *
from ovis.models.distributions import *
from ..models import Baseline
from ..models import VAE


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
