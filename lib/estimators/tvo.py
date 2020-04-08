from .base import *


class ThermoVariationalObjective(Estimator):
    """
    Thermovariational Inference. Based on https://arxiv.org/pdf/1907.00031.pdf / https://github.com/vmasrani/tvo
    Currently implemented for discrete models.

    TODO: Continuous models?
    """

    def compute_loss(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor], partition=1,
                     integration='left') -> Dict[str, Tensor]:
        """
        TODO: ADD PARTITION ARGUMENT

        Computes the covariance gradient estimator for the TVO bound.

         TVO = Delta_beto_0 ELBO + sum_{i=1..K-1} Delta_beta_k E_{pi_{beta_k]}}( f(x,z) )], f(x,z) = log p(x,z) - log q(z | x)

         Gradient estimator, eq. 11:
         Nabla E_{pi_{beta_k]}}( f(x,z) )] = E_{pi_{beta_k]}}( Nabla f(x,z) )] + Cov( Nabla log tilde{pi}(z), f(z) )

         (((In this expression the KLs are concatenated by stochastic layer so the freebits can be applied to each of them.)))

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
        :param num_particles: int; number of samples S in importance sampling of expectations
        :param integration: type of integral approximation (Riemann sum); left, right or trapz
        :return: dictionary with outputs [tvo, elbo, kl, log_f_x,z, N_eff]
        """

        # Generalize to accept custom partitions (current partition taken from default TVO settings on github)
        partition = torch.FloatTensor([0.0000e+00, 1.0000e-10, 1.2915e-09, 1.6681e-08, 2.1544e-07, 2.7826e-06,
                                       3.5938e-05, 4.6416e-04, 5.9948e-03, 7.7426e-02, 1.0000e+00]).cuda()

        num_particles = self.iw
        bs = self.bs  # temporary

        # compute the effective sample size
        N_eff = self.effective_sample_size(log_pzs, log_qzs)

        def cat_by_layer(log_pzs):
            """sum over the latent dimension (N_i) and concatenate stocastic layers.
            for L=2, a list of values [[*, N_1], [*, N_2]] becomes [*, 2]"""
            return torch.cat([x.sum(1, keepdims=True) for x in log_pzs], 1)

        # kl = E_q[ log p(z) - log q(z) ]
        log_pz = cat_by_layer(log_pzs)
        log_qz = cat_by_layer(log_qzs)
        if self.detach_qlogits:
            log_qz = log_qz.detach()
        kl = log_qz - log_pz

        # freebits is ditributed equally over the last dimension
        # (meaning L layers result in a total of L * freebits budget)
        kl = self.freebits(kl.unsqueeze(-1))
        kl = batch_reduce(kl)

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # compute log of weights w_s = p(x,z)/q(z|x) (between eq. 13 and 14)
        log_weight = log_f_xz.view(bs, num_particles)  # log_p_xz - log_q
        heated_log_weight = log_weight.unsqueeze(-1) * partition

        def exponentiate_and_normalize(values, dim=0):
            """Exponentiates and normalizes a tensor.
            Args:
                values: tensor [dim_1, ..., dim_N]
                dim: n
            Returns:
                result: tensor [dim_1, ..., dim_N]
                    where result[i_1, ..., i_N] =
                                    exp(values[i_1, ..., i_N])
                    ------------------------------------------------------------
                     sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N]) """
            log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
            return torch.exp(values - log_denominator)

        # compute bar{w^\beta_s} (used in eq. 13)
        heated_normalized_weight = exponentiate_and_normalize(heated_log_weight, dim=1)

        # compute tilde{pi}_beta(z) (eq. 7)
        log_p = log_px_z.view(-1, 1) + log_pz  # <---- is this correct ?
        log_p = log_p.view(bs, num_particles)
        log_qz = log_qz.view(bs, num_particles)
        thermo_logp = partition * log_p.unsqueeze(-1) + (1 - partition) * log_qz.unsqueeze(-1)

        # compute product in eq. 13 with f(x,z) = log[p(x,z)/q(z|x)]
        wf = heated_normalized_weight * log_weight.unsqueeze(-1)
        w_detached = heated_normalized_weight.detach()

        # num_particles S for importance sampling as
        # described in subsection Expectations of Section 4
        if num_particles == 1:
            correction = 1
        else:
            correction = num_particles / (num_particles - 1)

        # compute covariance (eq. 12)
        # .detach() makes sure PyTorch does not differentiate f_lambda(z) term.
        cov_term = correction * torch.sum(w_detached * (log_weight.unsqueeze(-1) -
                                                        torch.sum(wf, dim=1, keepdim=True)).detach() * (thermo_logp -
                                                                                                        torch.sum(
                                                                                                            thermo_logp * w_detached,
                                                                                                            dim=1,
                                                                                                            keepdim=True)),
                                          dim=1)

        # compute distances of partioning
        multiplier = torch.zeros_like(partition)
        if integration == 'trapz':
            multiplier[0] = 0.5 * (partition[1] - partition[0])
            multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
            multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
        elif integration == 'left':
            multiplier[:-1] = partition[1:] - partition[:-1]
        elif integration == 'right':
            multiplier[1:] = partition[1:] - partition[:-1]

        # compute covariance gradient estimator (eq. 11 / appendix F)

        """ Take mean over samples """
        # tvo = torch.mean(torch.sum(
        # multiplier * (thing_to_add + torch.sum(w_detached * log_weight.unsqueeze(-1),
        # dim=1)), dim=1))

        """ No mean over samples (necessary in this setup) """
        tvo = torch.sum(
            multiplier * (cov_term + torch.sum(w_detached * log_weight.unsqueeze(-1),
                                               dim=1)), dim=1)

        # multisampling ELBO
        log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
        elbo = torch.mean(log_evidence)

        return {'tvo': tvo, 'elbo': elbo, 'kl': kl, 'log_f_xz': log_f_xz, 'N_eff': N_eff}

    def effective_sample_size(self, log_pz, log_qz):

        # Not changed anything for TVO

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

        # compute effective sample size
        if self.iw > 1:
            # compute effective sample size
            log_w = batch_reduce(log_pz - log_qz).view(-1, self.mc, self.iw)
            N_eff = torch.exp(2 * torch.logsumexp(log_w, dim=2) - torch.logsumexp(2 * log_w, dim=2))
            N_eff = N_eff.mean(1)  # MC
        else:
            x = (log_pz).view(-1, self.mc, self.iw)
            N_eff = torch.ones_like(x[:, 0, 0])

        return N_eff

    def evaluate_model(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Dict[str, Tensor]:

        # Not changed anything for TVO

        if self.sequential_computation:
            # todo: move this in the forward pass so we retain only point estimates (loop over `evaluate model` method with iw = 1)
            bs, *dims = x.size()
            log_px_zs = []
            log_pzs = []
            log_qzs = []

            for i in range(self.iw):
                __x = x[:, None].repeat(1, self.mc, *(1 for _ in dims)).view(-1, *dims)

                # forward pass
                output = model(__x, **kwargs)
                px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]

                # compute log p(x|z), log p(z) and log q(z | x)
                log_px_zs += [batch_reduce(px.log_prob(__x))]
                log_pzs += [[pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]]
                log_qzs += [[qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]]

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

            # for now this does not return the model output as it is complicated to concatenate all of them (distributions)
            # this is why this should be used only for evaluation using basic VI
            return {'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz}

        else:
            # expand input to get MC and IW samples: [bs, mc, iw, ...]
            __x = self._expand_sample(x)

            # forward pass
            output = model(__x, **kwargs)
            px, z, qz, pz = [output[k] for k in ['px', 'z', 'qz', 'pz']]
            # compute log p(x|z), log p(z) and log q(z | x)
            log_px_z = batch_reduce(px.log_prob(__x))
            log_pz = [pz_l.log_prob(z_l) for pz_l, z_l in zip(pz, z)]
            log_qz = [qz_l.log_prob(z_l) for qz_l, z_l in zip(qz, z)]

            output.update({'log_px_z': log_px_z, 'log_pz': log_pz, 'log_qz': log_qz})
            return output

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:

        # Removed '.mean(1)'s and changed namings for TVO

        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        tvo_data = self.compute_loss(log_px_z, log_pz, log_qz)

        # loss
        tvo = tvo_data.get('tvo')  # .mean(1)  # MC averaging (done in loss above - move here?)
        loss = - tvo

        elbo = tvo_data.get('elbo')  # .mean(1)  # MC averaging (done in loss above - move here?)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': elbo,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(tvo_data.get('kl')),
                     'r_eff': tvo_data.get('N_eff') / self.iw},
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
