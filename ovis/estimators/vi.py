from torch.distributions import Distribution

from .base import *
from ..utils.utils import batch_reduce


class VariationalInference(Estimator):
    """
    Base class for Variational Inference using the Importance Weighted Bound (IWAE).
    Using this class to estimate the gradients requires the model to be compatible with the reparametrization trick.
    However the class can be used independently to evaluate the bound L_k.
    """

    @staticmethod
    def compute_iw_bound(log_px_z: Tensor, log_pz: Tensor, log_qz: Tensor,
                         detach_qlogits: bool = False, beta: float = 1.0, alpha: float = 0,
                         freebits: Optional[float] = None, request: List[str] = list(), **kwargs) -> Dict[str, Tensor]:
        """
        Compute the importance weighted bound for K = self.iw:

          * L_k = E_{q(z^1...z^K | x)} [ \log Z]
          * Z = 1/K \sum_{k=1..K} w_k
          * w_k  = p(x,z^k) / q(z^k|x)

        When using `alpha` > 0, the computed bound is the Importance Rényi Bound (IWR) given by:

          * L_k^\alpha = E_{q(z^1...z^K | x)} [ 1/(1-\alpha) \log Z(alpha)]
          * Z(alpha) = 1/K \sum_{i=1..K} w_k^{1 - \alpha}

        :param log_px_z: log p(x | z) of shape [bs, mc, iw]
        :param log_pz: `log p(z) l=1..L]` of shape [bs, mc, iw, L]
        :param log_qz: `log q(z|x) l=1..L]` of shape [bs, mc, iw, L]
        :param detach_qlogits: detach `log q(z|x)` to prevent getting gradients through `phi`
        :param beta: weight for the KL term (i.e. Beta-VAE [https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf])
        :param alpha: parameter of the IWR bound
        :params freebits: number of freebits applied to each layer
        :param request: return additional variables in ['log_pz', 'log_qz', 'log_px_z']
        :return: dictionary with keys [L_k, elbo, kl, log_wk] + keys stated in `request`
        """

        # detach log q(z^k | x) in case L_k should only be differentiable with regards to `theta`.
        if detach_qlogits:
            log_qz = log_qz.detach()

        # kl = E[log q(z^k|x) | p(z^k)]
        kl = log_qz - log_pz

        # apply Freebits
        if freebits is not None:
            kl = VariationalInference.apply_freebits(kl, freebits)

        # Sum the last dimensions (groups, or number of stochastic layers)
        kl = kl.sum(-1)

        # compute log w_k = log p(x, z^k) - log q(z^k | x) (ELBO)
        log_wk = log_px_z - beta * kl

        # log w_k^gamma = gamma * log w_k
        log_wk = (1 - alpha) * log_wk

        # L_k^alpha (IWR bound, IW bound when alpha = 0)
        L_k = torch.logsumexp(log_wk, dim=2) - VariationalInference.log_k(log_wk)
        L_k = L_k / (1. - alpha)

        # elbo
        elbo = torch.mean(log_wk, dim=(1, 2))

        # kl(q(z|x) || p(z|x)) = \log \hat{p} - L_1, \log \hat{p} = L_K,  accurate when K -> \inf
        kl_q_p = L_k.mean(1) - elbo

        # compute the effective sample size
        ess = VariationalInference.effective_sample_size(log_wk)

        output = {'log_wk': log_wk,
                  'L_k': L_k,
                  'elbo': elbo,
                  'ess': ess,
                  'kl_q_p': kl_q_p,
                  'kl': kl}

        if len(request):  # append additional data to the output
            meta = {'log_pz': log_pz,
                    'log_qz': log_qz,
                    'log_px_z': log_px_z}

            output.update(**{k: v for k, v in meta.items() if k in request})

        return output

    @staticmethod
    def log_k(log_wk: Tensor):
        return torch.tensor(log_wk.shape[2], device=log_wk.device, dtype=log_wk.dtype).log()

    @staticmethod
    def apply_freebits(kls: Tensor, value: float):
        """
        Apply `value` freebits [https://arxiv.org/abs/1606.04934] to each group.
        warnings: this implementation differs from the original implementation as freebits are here applied independently
        to each datapoint.
        :param kls: KL tensor where each element of the last dimension corresponds to one group
        :param value: Freebits value applied to each group
        :return: min(value, kls)
        """
        return kls.clamp(min=value)

    @staticmethod
    @torch.no_grad()
    def effective_sample_size(log_w):
        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2 from the log weights log w.
        :param log_w: log weights
        :return: effective sample size of shape [bs,]
        """
        if log_w.size(2) > 1:
            # computation in log-space
            a = 2 * torch.logsumexp(log_w, dim=2)
            b = torch.logsumexp(2 * log_w, dim=2)
            ess = torch.exp(a - b)
        else:
            ess = torch.ones_like(log_w[:, :, 0])

        return ess.mean(1)

    @staticmethod
    def evaluate_model(model: nn.Module, x: Tensor, iw: int = 1, mc: int = 1, **kwargs: Any) -> Dict[str, Tensor]:
        """
        Perform a forward pass through the `model` given the observation `x` and return the log probabilities, all of
        shape [bs, mc, iw, *]
        :param model: VAE model
        :param x: observation
        :param mc: number of outer Monte-Carlo samples
        :param iw: number of Importance Weighted samples
        :param kwargs: parameters for the model
        :return: {'log_px_z' : log p(x|z), 'log_qz' : log q(z|x), 'log_pz': log p(z), **model_output}
        """
        bs, *dims = x.size()

        # expand x as shape [bs * mc * iw, *dims]
        x_expanded = Estimator._expand_sample(x, mc, iw)

        # forward pass
        output = model(x_expanded, **kwargs)

        # unpack the model's output
        # the output must contain
        # a. px = p(x|z): torch.distributions.Distribution instance
        # b. z = latent sample z: List[Tensor]
        # c. p(z) = prior: List[torch.distributions.Distribution]
        # d. q(z | x) = approximate posterior: List[torch.distributions.Distribution]
        # a,b,c are lists where each element represents a stochastic layer
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # evaluate the log-densities given the observation `x` and the latent sample `z`
        # i.e. compute: log p(x|z), log p(z) and log q(z|x)
        # n.b. all log probabilities are of shape [bs * mc * iw,]
        log_px_z = batch_reduce(px.log_prob(x_expanded))
        log_pz = VariationalInference.eval_and_concat_log_probs(pz, z)
        log_qz = VariationalInference.eval_and_concat_log_probs(qz, z)

        # update the model's output with the log-densities
        output.update({'log_px_z': log_px_z.view(bs, mc, iw),
                       'log_pz': log_pz.view(bs, mc, iw, -1),
                       'log_qz': log_qz.view(bs, mc, iw, -1)})
        return output

    @staticmethod
    def eval_and_concat_log_probs(pz: List[Distribution], z: List[Tensor]):
        """
        Evaluate the list of log probabilities and concatenate.
        :param pz: list of `L` distributions `p(z)` of shape [*, ...]
        :param z: list of `L` samples `z` of shape [*, ...]
        :return: `log p(z)` of shape [*, L]
        """
        assert len(pz) == len(z)
        log_pzs = (batch_reduce(pz_l.log_prob(z_l)) for pz_l, z_l in zip(pz, z))
        return torch.cat([batch_reduce(x)[:, None] for x in log_pzs], 1)

    def sequential_evaluation(self, model: nn.Module, x: Tensor, mc: int = 1, iw: int = 1, **kwargs: Any):
        """
        Same as `evaluate_model` however the processing of the importance weighted samples
        is performed sequentially instead of in parallel.
        Warning: Except for the log-probabilites the model's output is only returned for one sample
        :param model: VAE model
        :param x: observation
        :param mc: number of outer Monte-Carlo samples
        :param iw: number of Importance Weighted samples
        :param kwargs: parameters for the model
        :return: {'log_px_z' : log p(x|z), 'log_qz' : log q(z|x), 'log_pz': log p(z), **model_output[-1]}
        """
        log_px_z = None
        log_pz = None
        log_qz = None
        for i in range(max(1, iw)):
            # evaluate batch
            output = self.evaluate_model(model, x, mc, 1, **kwargs)
            log_px_z_i, log_pz_i, log_qz_i = [output[k] for k in ['log_px_z', 'log_pz', 'log_qz']]

            # append log probs
            if log_px_z is None:
                log_px_z = log_px_z_i
                log_pz = log_pz_i
                log_qz = log_qz_i
            else:
                log_px_z = torch.cat([log_px_z, log_px_z_i], 2)
                log_pz = torch.cat([log_pz, log_pz_i], 2)
                log_qz = torch.cat([log_qz, log_qz_i], 2)

        # updatwe the output
        output.update({'log_px_z': log_px_z,
                       'log_pz': log_pz,
                       'log_qz': log_qz})

        return output

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, beta: float = 1.0,
                return_diagnostics: bool = True, **kwargs: Any) -> Tuple[
        Tensor, Diagnostic, Dict]:
        """
        Perform a forward pass through the VAE model, evaluate the Importance-Weighted bound and [optional] perform the backward pass.
        See the method `compute_iw_bound` for further documentation
        :param model: VAE model
        :param x: observation
        :param backward: perform the backward pass by calling loss.backward()
        :param beta: beta parameter for Beta-VAE [https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf]
        :param kwargs: additional arguments for the model's forward pass and the method `compute_iw_bound`
        :return: (loss : Tensor of shape [bs,],
                  diagnostics: nested dictionary containing at least the keys {'loss':{'loss':..., 'elbo':...}}
                  output: model's output + meta data)
        """

        config = self.get_runtime_config(**kwargs)

        if config.get('sequential_computation', False):
            # warning: the output will correspond to a single `iw` sample
            output = self.sequential_evaluation(model, x, **config)
        else:
            output = self.evaluate_model(model, x, **config)

        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, **kwargs)

        # compute the loss = - L_k  (backpropagation requires the reparametrization trick)
        L_k = iw_data.get('L_k').mean(1)  # 1/M \sum_m l_K[m] where M = mc
        loss = - L_k.mean()

        # prepare diagnostics
        diagnostics = Diagnostic()

        if return_diagnostics:
            diagnostics.update({
                'loss': {'loss': loss,
                         **self.base_loss_diagnostics(**iw_data, **output)}
            })

            # add auxiliary diagnostics that can customized through the method `additional_diagnostics`
            diagnostics.update(self.additional_diagnostics(**output))

        if backward:
            loss.backward()

        return loss, diagnostics, output

    @staticmethod
    def base_loss_diagnostics(**output):
        output.update(nll=- output['log_px_z'])
        keys = ['L_k', 'elbo', 'kl_q_p', 'ess', 'kl', 'nll']
        return {k: v.view(v.size(0), -1).mean(1) for k, v in output.items() if k in keys}

    @staticmethod
    @torch.no_grad()
    def additional_diagnostics(**output):
        """A function to append additional diagnostics from the model otuput"""
        gmm = {}
        if 'posterior_mse' in output.keys():
            gmm['posterior_mse'] = output['posterior_mse']
        if 'prior_mse' in output.keys():
            gmm['prior_mse'] = output['prior_mse']

        loss = {}
        if 'inferred_n' in output.keys():
            loss['inferred_n'] = output['inferred_n']

        gaussian_toy = {}
        for key in ['mse_A', 'mse_b', 'mse_mu', 'mse_phi']:
            if key in output.keys():
                gaussian_toy[key] = output[key]

        return {'gmm': gmm, 'loss': loss, 'gaussian_toy': gaussian_toy}


class PathwiseVAE(VariationalInference):

    def __init__(self, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(mc=iw * mc, iw=1, **kwargs)


class PathwiseIWAE(VariationalInference):

    def __init__(self, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(mc=1, iw=iw * mc, **kwargs)
