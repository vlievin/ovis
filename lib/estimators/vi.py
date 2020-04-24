from .base import *


class VariationalInference(Estimator):
    """
    Variational Inference. Using this estimator requires the model to be compatible with the reparametrization trick.

    """

    def compute_iw_bound(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor],
                         detach_qlogits: bool = False, request: List[str] = list()) -> Dict[str, Tensor]:
        """
        Compute the importance weighted bound:

         L_k = E_{q(z^1...z^K | x)} [ log 1/K \sum_{i=1..K} w_k], w_k  = p(x,z^k) / q(z^k|x)

         In this expression the KLs are concatenated by stochastic layer so the freebits can be applied to each of them.

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param detach_qlogits: detach the logits of q(z|x)
        :param request: list of variables to return
        :return: dictionary with outputs [L_k, kl, log_wk]
        """

        # compute the effective sample size
        N_eff = self.effective_sample_size(log_px_z, log_pzs, log_qzs)

        def cat_by_layer(log_pzs):
            """sum over the latent dimension (N_i) and concatenate stocastic layers.
            for L=2, a list of values [[*, N_1], [*, N_2]] becomes [*, 2]"""
            return torch.cat([x.sum(1, keepdims=True) for x in log_pzs], 1)

        # kl = E_q[ log p(z) - log q(z) ]
        log_pz = cat_by_layer(log_pzs)
        log_qz = cat_by_layer(log_qzs)
        if detach_qlogits:
            log_qz = log_qz.detach()
        kl = log_qz - log_pz
        # freebits is ditributed equally over the last dimension
        # (meaning L layers result in a total of L * freebits budget)
        if self.freebits is not None:
            kl = self.freebits(kl.unsqueeze(-1))
        kl = batch_reduce(kl)

        # compute log w_k = log p(x, z^k) - log q(z^k | x) (ELBO)
        log_wk = log_px_z - kl

        # view log_wk as shape [bs, mc, iw]
        log_wk = log_wk.view(-1, self.mc, self.iw)

        # IW-ELBO: L_k
        L_k = torch.logsumexp(log_wk, dim=2) - self.log_iw  # if self.iw > 1 else log_f_xz.squeeze(2)

        output = {'L_k': L_k, 'kl': kl, 'log_wk': log_wk, 'N_eff': N_eff}

        if len(request):
            # append additional data to the output
            meta = {'log_pz': log_pz, 'log_qz': log_qz, 'log_px_z': log_px_z}
            for key in request:
                output[key] = batch_reduce(meta[key]).view(-1, self.mc, self.iw)

        return output

    @torch.no_grad()
    def effective_sample_size(self, log_px_z, log_pz, log_qz):
        """
        Compute the effective sample size: N_eff = (\sum_i w_i)**2 / \sum_i w_i**2
        :param log_pz: log p(z) of shape [bs * mc * iw, ...]
        :param log_qz: loq q(z) of shape [bs * mc * iw, ...]
        :return: effective_sample_size
        """

        if isinstance(log_pz, List):
            log_pz = torch.cat(log_pz, 1)

        if isinstance(log_qz, List):
            log_qz = torch.cat(log_qz, 1)

        # sum over dimensions of z
        log_pz = log_pz.sum(1)
        log_qz = log_qz.sum(1)

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            log_w = batch_reduce(log_px_z + log_pz - log_qz).view(-1, self.mc, self.iw)
            N_eff = torch.exp(2 * torch.logsumexp(log_w, dim=2) - torch.logsumexp(2 * log_w, dim=2))
            N_eff = N_eff.mean(1)  # MC
        else:
            x = (log_pz).view(-1, self.mc, self.iw)
            N_eff = torch.ones_like(x[:, 0, 0])

        return N_eff

    def evaluate_model(self, model: nn.Module, x: Tensor, x_target: Tensor, **kwargs: Any) -> Dict[str, Tensor]:
        # forward pass
        output = model(x, **kwargs)
        px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

        # compute log p(x|z), log p(z) and log q(z | x)
        log_px_z = batch_reduce(px.log_prob(x_target))
        log_pz = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
        log_qz = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

        output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})
        return output

    def _sequential_evaluation(self, model: nn.Module, x: Tensor, **kwargs: Any):
        bs, *dims = x.size()
        log_px_zs = []
        log_pzs = []
        log_qzs = []
        x_target = x[:, None].repeat(1, self.mc, *(1 for _ in dims)).view(-1, *dims)
        for i in range(self.iw):
            # evaluate batch
            output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=1, **kwargs)
            log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]

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

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        if self.sequential_computation:
            # warning: here only one `output` will be returned
            output = self._sequential_evaluation(model, x, **kwargs)
        else:
            x_target = self._expand_sample(x)
            output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)

        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz)

        # loss
        L_k = iw_data.get('L_k').mean(1)  # MC averaging
        loss = - L_k

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(iw_data.get('kl')),
                     'r_eff': iw_data.get('N_eff') / self.iw},
        })

        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output

    def _diagnostics(self, output):
        """A function to append additional diagnostics from the model otuput"""

        prior = {}
        if 'Hp' in output.keys():
            prior['hp'] = torch.sum(torch.cat(output['Hp']))
        if 'usage' in output.keys():
            prior['usage'] = torch.mean(torch.cat(output['usage']))

        gmm = {}
        if 'posterior_mse' in output.keys():
            gmm['posterior_mse'] = output['posterior_mse']
        if 'prior_mse' in output.keys():
            gmm['prior_mse'] = output['prior_mse']

        return {'prior': prior, 'gmm': gmm}


class PathwiseVAE(VariationalInference):

    def __init__(self, beta: float = 1, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(beta=beta, mc=iw * mc, iw=1, **kwargs)


class PathwiseIWAE(VariationalInference):

    def __init__(self, beta: float = 1, mc: int = 1, iw: int = 1, **kwargs):
        super().__init__(beta=beta, mc=1, iw=iw * mc, **kwargs)


class SafeVariationalInference(VariationalInference):
    """A Variational Inference class without bells and whistles for debugging purposes"""

    def _expand_sample(self, x):
        bs, *dims = x.size()
        self.bs = bs  # added for TVO - perhaps a more elegant fix is possible.
        x = x[:, None, None].repeat(1, self.mc, self.iw, *(1 for _ in dims))
        # flatten everything into the batch dimension
        return x.view(-1, *dims)

    def _reduce_sample(self, x):
        _, *dims = x.size()
        x = x.view(-1, self.mc, self.iw, *dims)
        return x.mean((1, 2,))

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
        bs = x.size(0)
        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]

        assert len(log_pz) == 1
        log_pz = batch_reduce(log_pz[0])
        log_qz = batch_reduce(log_qz[0])

        log_wk = log_px_z + log_pz - log_qz
        log_wk = log_wk.view(bs, self.mc, self.iw)

        # compute IW-bound
        L_k = torch.logsumexp(log_wk, dim=2).mean(1)

        # compute stats
        N_eff = (log_wk.exp().sum(2) ** 2 / (log_wk.exp() ** 2).sum(2)).mean(1)
        kl = (log_pz - log_qz).view(bs, self.mc, self.iw).mean(dim=(1, 2))
        log_px_z = log_px_z.view(bs, self.mc, self.iw).mean(dim=(1, 2))

        # loss
        loss = - L_k

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': L_k,
                     'nll': - log_px_z,
                     'kl': kl,
                     'r_eff': N_eff / self.iw},
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
