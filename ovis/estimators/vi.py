from torch.distributions import Distribution

from .base import *
from ..models import TemplateModel
from ..utils.utils import batch_reduce, batch_average


class VariationalInference(GradientEstimator):
    """
    Base class for Variational Inference using the Importance Weighted Bound (IWAE).
    Using this class to estimate the gradients requires the model to be compatible with the reparametrization trick.
    This class can be used independently to evaluate the bound L_k.
    """

    def forward(self,
                model: nn.Module,
                x: Tensor, backward: bool = False,
                return_diagnostics: bool = True,
                **kwargs: Any) -> Tuple[Tensor, Diagnostic, Dict]:
        """
        Perform a forward pass through the VAE model, evaluate the Importance-Weighted bound and [optional] perform the backward pass.
        See the method `compute_iw_bound` for further documentation.
        :param model: VAE model
        :param x: observation
        :param backward: perform the backward pass by calling loss.backward()
        :param kwargs: additional arguments for the model's forward pass and the method `compute_iw_bound`
        :return: (loss : Tensor of shape [bs,],
                  diagnostics: nested dictionary containing at least the keys {'loss':{'loss':..., 'elbo':...}}
                  output: model's output + meta data)
        """

        # update the `config` object with the `kwargs`
        config = self.get_runtime_config(**kwargs)

        # forward pass and eval of the log probs
        if config.get('sequential_computation', False):
            log_probs, output = self.sequential_evaluation(model, x, **config)
        else:
            log_probs, output = self.evaluate_model(model, x, **config)

        iw_data = self.compute_log_weights_and_iw_bound(**log_probs, **config)

        # compute the loss = - L_k  (backpropagation requires the reparametrization trick)
        L_k = iw_data.get('L_k').mean(1)  # 1/M \sum_m l_K[m] where M = mc
        loss = - L_k.mean()

        # prepare diagnostics
        diagnostics = Diagnostic()
        if return_diagnostics:
            diagnostics = self.base_loss_diagnostics(**iw_data, **log_probs)
            diagnostics.update(self.additional_diagnostics(**output))

        if backward:
            loss.backward()

        return loss, diagnostics, output

    @staticmethod
    def evaluate_model(model: TemplateModel, x: Tensor, iw: int = None, mc: int = None, **kwargs: Any) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the `model` given the observation `x` and return the log probabilities, all of
        shape [bs, mc, iw, *]
        :param model: VAE model
        :param x: observation
        :param mc: number of outer Monte-Carlo samples
        :param iw: number of Importance Weighted samples
        :param kwargs: parameters for the model
        :return: ({'log_px_z' : log p(x|z), 'log_qz' : log q(z|x), 'log_pz': log p(z)}, model_output, )
        """
        bs, *dims = x.size()

        # expand x as shape [bs * mc * iw, *dims]
        x_expanded = GradientEstimator._expand_sample(x, mc, iw)

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
        log_probs = {'log_px_z': log_px_z.view(bs, mc, iw),
                     'log_pz': log_pz.view(bs, mc, iw, -1),
                     'log_qz': log_qz.view(bs, mc, iw, -1)}

        return log_probs, output

    @staticmethod
    def compute_log_weights(log_px_z: Tensor = None,
                            log_pz: Tensor = None,
                            log_qz: Tensor = None,
                            detach_qlogits: bool = False,
                            beta: float = 1.0,
                            alpha: float = 0,
                            freebits: Optional[float] = None,
                            **kwargs: Any) -> Tensor:
        """
        Compute the log weights from the log probs `p(x|z)`, `p(z)` and `q(z|x)`:

          * log w_k = log p(x,z) - log q(z|x)

        :param log_px_z: log p(x | z) of shape [bs, mc, iw]
        :param log_pz: `log p(z) l=1..L]` of shape [bs, mc, iw, L]
        :param log_qz: `log q(z|x) l=1..L]` of shape [bs, mc, iw, L]
        :param detach_qlogits: detach `log q(z|x)` to prevent getting gradients through `phi`
        :param beta: weight for the KL term (i.e. Beta-VAE [https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf])
        :param alpha: parameter of the IWR bound
        :params freebits: number of freebits applied to each layer
        :return: a tuple log w_k
        """

        # detach log q(z^k | x) in case L_k should only be differentiable with regards to `theta`.
        if detach_qlogits:
            log_qz = log_qz.detach()

        # kl = E[log q(z^k|x) - log p(z^k)]
        kl = log_qz - log_pz

        # apply Freebits
        if freebits is not None:
            kl = VariationalInference.apply_freebits(kl, freebits)

        # Sum the last dimensions (groups, or number of stochastic layers)
        kl = kl.sum(-1)

        # compute log w_k = log p(x, z^k) - log q(z^k | x) (ELBO)
        log_wk = log_px_z - beta * kl

        # log w_k^gamma = gamma * log w_k
        return (1 - alpha) * log_wk

    @staticmethod
    def compute_iw_bound(log_wk: Tensor = None,
                         alpha: float = 0,
                         **kwargs) -> Tensor:
        """
        Compute the importance weighted bound for `K = log_wk.shape[2]`:

          * L_k = E_{q(z^1...z^K | x)} [ \log Z]
          * Z = 1/K \sum_{k=1..K} w_k

        When using `alpha` > 0, the computed bound is the Importance Rényi Bound (IWR) given by:

          * L_k^\alpha = E_{q(z^1...z^K | x)} [ 1/(1-\alpha) \log Z(alpha)]
          * Z(alpha) = 1/K \sum_{i=1..K} w_k^{1 - \alpha}

        :param log_wk: log weights `log w_k`
        :param alpha: Renyi Bound parameter (alpha=0 <=> Importance Weighted Bound)
        :param kwargs: additional params
        :return: L_k
        """
        L_k = log_wk.logsumexp(dim=-1) - VariationalInference.cast_log(log_wk.shape[-1], log_wk)
        return L_k / (1. - alpha)

    @staticmethod
    def compute_log_weights_and_iw_bound(**kwargs) -> Dict[str, Tensor]:
        """
        Compute the importance weighted bound for `K = log_pz.shape[2]`:

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
        :return: dictionary with keys [L_k, elbo, kl, log_wk] + keys stated in `request`
        """

        # compute the log weights + kl
        log_wk = VariationalInference.compute_log_weights(**kwargs)

        # L_k^alpha (IWR bound, IW bound when alpha = 0)
        L_k = VariationalInference.compute_iw_bound(log_wk=log_wk, **kwargs)

        return {'log_wk': log_wk, 'L_k': L_k}

    @staticmethod
    def cast_log(value: int, ref_tensor: Tensor):
        """cast `value` to `ref_tensor` dtype and return the `log(value)`"""
        return torch.tensor(value, device=ref_tensor.device, dtype=ref_tensor.dtype, requires_grad=False).log()

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

    def sequential_evaluation(self, model: TemplateModel, x: Tensor, mc: int = None, iw: int = None, **kwargs: Any) -> \
            Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Same as `evaluate_model` however the processing of the importance weighted samples
        is performed sequentially instead of in parallel.
        **Warning**: Except for the log-probabilites the model's output is only returned for one sample
        :param model: VAE model
        :param x: observation
        :param mc: number of outer Monte-Carlo samples
        :param iw: number of Importance Weighted samples
        :param kwargs: parameters for the model
        :return: ({'log_px_z' : log p(x|z), 'log_qz' : log q(z|x), 'log_pz': log p(z)}, model_output[-1], )
        """

        log_px_z = None
        log_pz = None
        log_qz = None
        for i in range(max(1, iw)):
            # evaluate mini-batch [bs, mc,]
            log_probs_i, output_i = self.evaluate_model(model, x, mc, 1, **kwargs)
            log_px_z_i, log_pz_i, log_qz_i = [log_probs_i[k] for k in ['log_px_z', 'log_pz', 'log_qz']]

            # append log probs
            if log_px_z is None:
                log_px_z = log_px_z_i
                log_pz = log_pz_i
                log_qz = log_qz_i
            else:
                log_px_z = torch.cat([log_px_z, log_px_z_i], 2)
                log_pz = torch.cat([log_pz, log_pz_i], 2)
                log_qz = torch.cat([log_qz, log_qz_i], 2)

        log_probs = {'log_px_z': log_px_z,
                     'log_pz': log_pz,
                     'log_qz': log_qz}

        return log_probs, output_i

    @staticmethod
    @torch.no_grad()
    def base_loss_diagnostics(log_wk: Tensor = None,
                              L_k: Tensor = None,
                              log_px_z: Tensor = None,
                              **kwargs) -> Diagnostic:

        # ELBO decomposition
        L_k = batch_average(L_k)
        elbo = batch_average(log_wk)
        nll = - batch_average(log_px_z)
        kl = -elbo - nll

        # kl(q(z|x) || p(z|x)) = \log \hat{p} - L_1, \log \hat{p} = L_K,  accurate when K -> \inf
        kl_q_p = L_k - elbo

        # effective sample size
        ess = batch_average(VariationalInference.effective_sample_size(log_wk))

        return Diagnostic({'loss': {
            'L_k': L_k,
            'elbo': elbo,
            'kl_q_p': kl_q_p,
            'nll': nll,
            'kl': kl,
            'ess': ess
        }})

    @staticmethod
    @torch.no_grad()
    def additional_diagnostics(**output) -> Diagnostic:
        """A function to append additional diagnostics from the model output"""
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

        return Diagnostic({'gmm': gmm, 'loss': loss, 'gaussian_toy': gaussian_toy})


class Pathwise(VariationalInference):

    def __init__(self, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(reparam=True, mc=mc, iw=iw, **kwargs)


class PathwiseVAE(VariationalInference):

    def __init__(self, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(reparam=True, mc=iw * mc, iw=1, **kwargs)


class PathwiseIWAE(VariationalInference):

    def __init__(self, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(reparam=True, mc=1, iw=iw * mc, **kwargs)
