from .base import *
from ..utils.utils import batch_reduce


class VariationalInference(Estimator):
    """
    Base class for Variational Inference using the Importance Weighted Bound (IWAE).
    Using this class to estimate the gradients requires the model to be compatible with the reparametrization trick.
    However the class can be used independently to evaluate the bound L_k.
    """

    def compute_iw_bound(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor],
                         detach_qlogits: bool = False, beta: float = 1.0, alpha: float = 0,
                         request: List[str] = list(), **kwargs) -> Dict[str, Tensor]:
        """
        Compute the importance weighted bound for K = self.iw:

          * L_k = E_{q(z^1...z^K | x)} [ \log Z]
          * Z = 1/K \sum_{k=1..K} w_k
          * w_k  = p(x,z^k) / q(z^k|x)

        When using `alpha` > 0, the computed bound is the Importance Rényi Bound (IWR) given by:

          * L_k^\alpha = E_{q(z^1...z^K | x)} [ 1/(1-\alpha) \log Z(alpha)]
          * Z(alpha) = 1/K \sum_{i=1..K} w_k^{1 - \alpha}

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i] (one value per stochastic layer)
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i] (one value per stochastic layer)
        :param detach_qlogits: detach the logits of q(z|x)
        :param beta: weight for the KL term (i.e. Beta-VAE [https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf])
        :param alpha: parameter of the IWR bound
        :param request: return additional variables in ['log_pz', 'log_qz', 'log_px_z']
        :return: dictionary with keys [L_k, elbo, kl, log_wk] + keys stated in `request`
        """

        def cat_by_layer(log_pzs):
            """sum over the latent dimension (N_i) and concatenate stocastic layers.
            for L=2, a list of values [[*, N_1], [*, N_2]] becomes [*, 2]"""
            return torch.cat([batch_reduce(x)[:, None] for x in log_pzs], 1)

        # concatenate log probs from each layer
        log_pz = cat_by_layer(log_pzs)
        log_qz = cat_by_layer(log_qzs)

        # detach log q(z^k | x) in case L_k should only be differentiable with regards to `theta`.
        if detach_qlogits:
            log_qz = log_qz.detach()

        # kl = E[log q(z^k|x) | p(z^k)]
        kl = log_qz - log_pz

        # freebits is ditributed equally over the last dimension (here lastdim == 1)
        # meaning that a total of L * freebits will be applied to a kl tensor of shape [*, L]
        if self.freebits is not None:
            kl = self.freebits(kl.unsqueeze(-1))

        # dimension [bs * MC,* IW, ...] -> [bs * MC,* IW]
        kl = batch_reduce(kl)

        # compute log w_k = log p(x, z^k) - log q(z^k | x) (ELBO)
        log_wk = log_px_z - beta * kl

        # log w_k^gamma = gamma * log w_k
        log_wk = (1 - alpha) * log_wk

        # view tensors as shape]
        log_wk = log_wk.view(-1, self.mc, self.iw + self.auxiliary_samples)

        # separate w_k from auxiliary samples w^{(s)} (ovis)
        log_wk_aux = log_wk[:, :, self.iw:]
        log_wk = log_wk[:, :, :self.iw]

        # L_k^alpha (IWR bound, IW bound when alpha = 0)
        L_k = torch.logsumexp(log_wk, dim=2) - self.log_iw
        L_k = L_k / (1. - alpha)

        # elbo
        elbo = torch.mean(log_wk, dim=(1, 2))

        # kl(q(z|x) || p(z|x)) = \log \hat{p} - L_1, \log \hat{p} = L_K,  accurate when K -> \inf
        kl_q_p = L_k.mean(1) - elbo

        # compute the effective sample size
        ess = self.effective_sample_size(log_wk)

        def format_tensor(x):
            """reshape tensor as [batch_size, mc, iw+S, ...]"""
            x = batch_reduce(x)
            return x.view(-1, self.mc, self.iw + self.auxiliary_samples)

        output = {'L_k': L_k, 'elbo': elbo, 'log_wk': log_wk, 'ess': ess,
                  'kl_q_p': kl_q_p, 'log_wk_aux': log_wk_aux,
                  'kl': kl.view(-1, self.mc, self.iw + self.auxiliary_samples).mean(dim=(1, 2))}

        if len(request):  # append additional data to the output
            meta = {'log_pz': format_tensor(log_pz),
                    'log_qz': format_tensor(log_qz),
                    'log_px_z': format_tensor(log_px_z)}

            output.update(**{k: v for k, v in meta.items() if k in request})

        return output

    @torch.no_grad()
    def effective_sample_size_from_probs(self, log_px_z, log_pz, log_qz):
        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2
        :param log_pz: log p(z) of shape [bs * mc * iw, ...]
        :param log_qz: loq q(z) of shape [bs * mc * iw, ...]
        :return: effective sample size of shape [bs,]
        """

        if isinstance(log_pz, List):
            log_pz = torch.cat([l.view(l.size(0), -1) for l in log_pz], 1)

        if isinstance(log_qz, List):
            log_qz = torch.cat([l.view(l.size(0), -1) for l in log_qz], 1)

        # sum over dimensions of z
        log_pz = log_pz.sum(1)
        log_qz = log_qz.sum(1)

        # compute log weights
        log_w = batch_reduce(log_px_z + log_pz - log_qz).view(-1, self.mc, self.iw)

        return self.effective_sample_size(log_w)

    @torch.no_grad()
    def effective_sample_size(self, log_w):
        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2 from the log weights log w.
        :param log_w: log weights
        :return: effective sample size of shape [bs,]
        """
        if self.iw > 1:
            # computation in log-space
            a = 2 * torch.logsumexp(log_w, dim=2)
            b = torch.logsumexp(2 * log_w, dim=2)
            ess = torch.exp(a - b)
        else:
            x = (log_w).view(-1, self.mc, self.iw)
            ess = torch.ones_like(x[:, :, 0])

        return ess.mean(1)

    def evaluate_model(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Dict[str, Tensor]:
        """
        Perform a forward pass through the `model` given the observation `x` and return the log probabilities.
        :param model: VAE model
        :param x: observation
        :param kwargs: parameters for the model
        :return: {'log_px_z' : log p(x|z), 'log_qz' : log q(z|x), 'log_pz': log p(z), **model_output}
        """

        # expand x as shape [bs * mc * (iw+S)]
        x_expanded = self._expand_sample(x)

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
        log_px_z = batch_reduce(px.log_prob(x_expanded))
        log_pz = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
        log_qz = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

        # update the model's output with the log-densities
        output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})
        return output

    def sequential_evaluation(self, model: nn.Module, x: Tensor, **kwargs: Any):
        """
        Same as `evaluate_model` however the processing of the importance weighted samples
        is performed sequentially instead of in parallel.
        Warning: Except for the log-probabilites the model's output is only returned for one sample
        :param model: VAE model
        :param x: observation
        :param kwargs: parameters for the model
        :return: {'log_px_z' : log p(x|z), 'log_qz' : log q(z|x), 'log_pz': log p(z), **model_output[-1]}
        """
        bs, *dims = x.size()
        log_px_zs = []
        log_pzs = []
        log_qzs = []
        x_expanded_mc = x[:, None].repeat(1, self.mc, *(1 for _ in dims)).view(-1, *dims)
        for i in range(self.iw):
            # evaluate batch
            output = model(x_expanded_mc, **kwargs)
            px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]
            log_px_z = batch_reduce(px.log_prob(x_expanded_mc))
            log_pz = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
            log_qz = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

            log_px_zs += [log_px_z]
            log_pzs += [log_pz]
            log_qzs += [log_qz]

            # concatenate
        log_px_z = torch.cat([x.view(bs, self.mc, 1) for x in log_px_zs], 2)

        log_pzs = [list(x) for x in zip(*log_pzs)]
        log_pz = [torch.cat([x.view(bs, self.mc, 1, *x.size()[1:]) for x in log_pzs_l], 2) for log_pzs_l in log_pzs]

        log_qzs = [list(x) for x in zip(*log_qzs)]
        log_qz = [torch.cat([x.view(bs, self.mc, 1, *x.size()[1:]) for x in log_qzs_l], 2) for log_qzs_l in log_qzs]

        # re-flatten bs, mc, iw samples
        log_px_z = log_px_z.view(-1)
        log_pz = [x.view(-1, *x.size()[3:]) for x in log_pz]
        log_qz = [x.view(-1, *x.size()[3:]) for x in log_qz]

        output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})

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

        if self.sequential_computation:
            # warning: the output will correspond to a single `iw` sample
            output = self.sequential_evaluation(model, x, **kwargs)
        else:
            output = self.evaluate_model(model, x, **kwargs)

        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, **kwargs)

        # compute the loss = - L_k  (backpropagation requires the reparametrization trick)
        L_k = iw_data.get('L_k').mean(1)  # 1/M \sum_m l_K[m] where M = self.mc
        loss = - L_k

        # prepare diagnostics
        diagnostics = Diagnostic()

        if return_diagnostics:
            diagnostics.update({
                'loss': {'loss': loss,
                         **self.base_loss_diagnostics(**iw_data, **output)}
            })

            # add auxiliary diagnostics that can customized through the method `additional_diagnostics`
            diagnostics.update(self.additional_diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output

    def base_loss_diagnostics(self, **output):
        L_k = output.get('L_k').mean(1)
        elbo = output.get('elbo')
        kl_q_p = output.get('kl_q_p')
        ess = output.get('ess')
        nll = - self._reduce_sample(output.get('log_px_z'))
        kl = output.get('kl')
        return {'L_k': L_k, 'elbo': elbo, 'kl_q_p': kl_q_p,
                'ess': ess, 'r_ess': ess / self.iw, 'nll': nll, 'kl': kl}

    @torch.no_grad()
    def additional_diagnostics(self, output):
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
