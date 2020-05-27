from torch import softmax, sigmoid

from .base import _EPS
from .vi import *
from ..distributions import *
from ..models import Baseline
from ..models import VAE
from ..models import BernoulliToyModel


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
        self.K = K
        self.N = N
        # control variate model
        self.r_rho = Baseline(xdim=(N, K), nlayers=1, hdim=hdim)
        self.r_rho_nb = Baseline(xdim=(N, K), nlayers=1, hdim=hdim, batch_norm=False)

        if self.K == 1:
            self.register_buffer("log_tau", torch.log(0.5 * torch.ones((1, N))))
        else:
            self.register_buffer("log_tau", torch.log(0.5 * torch.ones((1, N, 1))))

        # self.log_tau = nn.Parameter(torch.log(0.5 * torch.ones((1, N, 1)))) # todo: learning the temperature causes memory allocation error

    @property
    def tau(self):
        return self.log_tau.exp()

    def sigma(self, z, tau):
        if self.K == 1:
            return sigmoid(z / tau)
        else:
            return softmax(z / tau, dim=-1)

    def sample_posterior(self, logits):
        # sample noise
        if self.K == 1:
            # Sampling of z, b, z_tilde for special case Bernoulli (as done in https://github.com/duvenaud/relax/blob/master/pytorch_toy.py)
            u = torch.empty_like(logits).uniform_()
            v = torch.empty_like(logits).uniform_()

            # sample z ~ p(z | \theta)
            z = logits + torch.log(u) - torch.log1p(-u) # reparametrizing samples, appendix B

            # b = H(z) <-> b ~ p(b | \theta) (Gumbel-Max trick)
            b = z.gt(0.).type_as(z)

            # sample z_tilde ~ p(z | b, \theta) (Appendix B)
            theta = torch.sigmoid(logits)
            v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
        else:
            # Sampling of z, b, z_tilde for Categorical
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

    def f(self, model, qz, pz, x, z, beta=1.0):
        px = model.generate(z)

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x))
        log_pz = pz.log_prob(z)
        log_qz = qz.log_prob(z)

        # kl = E[log q(z^k|x) | p(z^k)]
        kl = log_qz - log_pz

        # freebits is ditributed equally over the last dimension
        # (meaning L layers result in a total of L * freebits budget)
        if self.freebits is not None:
            kl = self.freebits(kl.unsqueeze(-1))

        # dimension [bs, MC, IW]
        kl = batch_reduce(kl)

        # compute log w_k = log p(x, z^k) - log q(z^k | x) (ELBO)
        log_wk = log_px_z - beta * kl

        return log_wk, kl, -log_px_z, px # elbo, kl, -log_px_z, px

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, beta: float = 1.0, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
        assert isinstance(model, VAE) or isinstance(model, BernoulliToyModel), "only implemented for basic VAE and Bernoulli Toy"

        # expand sample
        x = self._expand_sample(x)

        if isinstance(model, BernoulliToyModel):
            qlogits, output = model.infer(x)
            # qlogits = output['qlogits']
        else: # Maybe fix this by adding diagnostics output to VAE model
            qlogits = model.infer(x)
        qlogits.retain_grad()
        # sample b, z, z_tilde
        z, b, z_tilde = self.sample_posterior(qlogits)
        # p(z) and q(z|x)
        if self.K == 1:
            pz = Bernoulli(logits=model.prior)
            qz = Bernoulli(logits=qlogits)
        else:
            pz = PseudoCategorical(model.prior)
            qz = PseudoCategorical(qlogits)

        # compute f(x, b)
        f_b, kl, nll, px = self.f(model, qz, pz, x, b, beta=beta)

        # compute control variates
        sig_z = self.sigma(z, self.tau)

        f_z, *_ = self.f(model, qz, pz, x, sig_z, beta=beta)
        if self.K == 1: # Remove this conditional by doing it in init
            c_z = f_z + self.r_rho_nb(sig_z)
        else:
            c_z = f_z + self.r_rho(sig_z)
        sig_z_tilde = self.sigma(z_tilde, self.tau)

        f_z_tilde, *_ = self.f(model, qz, pz, x, sig_z_tilde, beta=beta)
        if self.K == 1: # Remove this conditional by doing it in init
            c_z_tilde = f_z_tilde + self.r_rho_nb(sig_z_tilde)
        else:
            c_z_tilde = f_z_tilde + self.r_rho(sig_z_tilde)

        # for debugging
        control_variate_mse = (c_z_tilde - f_b).pow(2).detach()

        # loss
        reinforce_loss = torch.sum((f_b - c_z_tilde).unsqueeze(1).detach() * qz.log_prob(b), 1)
        # reinforce_loss = torch.sum((f_b - c_z_tilde).detach() * qz.log_prob(b), 1)
        loss = - (reinforce_loss + c_z - c_z_tilde) # - (f_b + reinforce_loss + c_z - c_z_tilde)  # todo: check if f_b terms is correct (doesn't work without that)
        # view loss as shape [bs, mc, iw]
        loss = loss.view(-1, self.mc, self.iw)
        # IW-loss:
        loss = (torch.logsumexp(loss, dim=2) - self.log_iw).view(-1)

        def _reduce(x):
            _, *_dims = x.shape
            # x = x.view(-1, self.iw, *_dims)
            x = x.view(-1, self.mc, *_dims)
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
        if isinstance(model, BernoulliToyModel):
            diagnostics.update(self._diagnostics(output))

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
    

class Rebar(Relax):
    """
    Rebar: https://arxiv.org/abs/1703.07370

    NB: this implementation is probably only 90% correct.

    todo: learn tau (setting tau as a parameter causes malloc error)
    """

    def __init__(self, *args, N=8, K=8, hdim=32, **kwargs):
        super().__init__(*args, **kwargs, N=N, K=K, hdim=hdim)

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, beta: float = 1.0, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
        assert isinstance(model, VAE) or isinstance(model, BernoulliToyModel), "only implemented for basic VAE and Bernoulli Toy"

        # expand sample
        x = self._expand_sample(x)

        if isinstance(model, BernoulliToyModel):
            qlogits, output = model.infer(x)
            # qlogits = output['qlogits']
        else: # Maybe fix this by adding diagnostics output to VAE model
            qlogits = model.infer(x)
        qlogits.retain_grad()

        # sample b, z, z_tilde
        z, b, z_tilde = self.sample_posterior(qlogits)

        # p(z) and q(z|x)
        if self.K == 1:
            pz = Bernoulli(logits=model.prior)
            qz = Bernoulli(logits=qlogits)
        else:
            pz = PseudoCategorical(model.prior)
            qz = PseudoCategorical(qlogits)

        # compute f(x, b)
        f_b, kl, nll, px = self.f(model, qz, pz, x, b, beta=beta)

        # compute control variates
        sig_z = self.sigma(z, self.tau)
        f_z, *_ = self.f(model, qz, pz, x, sig_z, beta=beta)

        sig_z_tilde = self.sigma(z_tilde, self.tau)

        f_z_tilde, *_ = self.f(model, qz, pz, x, sig_z_tilde, beta=beta)

        # for debugging
        control_variate_mse = (f_z_tilde - f_b).pow(2).detach()

        # loss
        reinforce_loss = torch.sum((f_b.detach() - model.eta * f_z_tilde.detach()).unsqueeze(1) * qz.log_prob(b), 1)
        loss = - (reinforce_loss + model.eta * (f_z - f_z_tilde)) # - (f_b + reinforce_loss + c_z - c_z_tilde)  # todo: check if f_b terms is correct (doesn't work without that)
        # view loss as shape [bs, mc, iw]
        loss = loss.view(-1, self.mc, self.iw)
        # IW-loss:
        loss = (torch.logsumexp(loss, dim=2) - self.log_iw).view(-1)

        def _reduce(x):
            _, *_dims = x.shape
            x = x.view(-1, self.mc, *_dims)
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
        diagnostics.update(self._diagnostics(output))

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
