from .vi import *

class ThermoVariationalObjective(VariationalInference):
    """
    Thermovariational Inference. Based on https://arxiv.org/pdf/1907.00031.pdf / https://github.com/vmasrani/tvo
    Currently implemed for discrete VAEs.

    TODO: Continuous VAEs?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Partitions of unit interval
        partitions = [[0.0000e+00, 1.0000e-10, 1.0000e+00],  # 0 (Poor)
                     [0.0000e+00, 3.0000e-10, 1.0000e+00],  # 1 (Poor)
                     [0.0000e+00, 1.0000e-9, 1.0000e+00],  # 2 (Poor)
                     [0.0000e+00, 3.0000e-9, 1.0000e+00],  # 3 (Poor)
                     [0.0000e+00, 1.0000e-8, 1.0000e+00],  # 4 (Poor)
                     [0.0000e+00, 3.0000e-8, 1.0000e+00],  # 5 (Poor)
                     [0.0000e+00, 1.0000e-7, 1.0000e+00],  # 6 (Poor)
                     [0.0000e+00, 3.0000e-7, 1.0000e+00],  # 7 (Poor)
                     [0.0000e+00, 1.0000e-6, 1.0000e+00],  # 8 (Poor)
                     [0.0000e+00, 3.0000e-6, 1.0000e+00],  # 9 (Poor)
                     [0.0000e+00, 1.0000e-5, 1.0000e+00],  # 10 (Poor)
                     [0.0000e+00, 3.0000e-5, 1.0000e+00],  # 11 (Poor)
                     [0.0000e+00, 1.0000e-4, 1.0000e+00],  # 12 (Poor)
                     [0.0000e+00, 3.0000e-4, 1.0000e+00],  # 13 (Poor)
                     [0.0000e+00, 1.0000e-3, 1.0000e+00],  # 14 (Poor)
                     [0.0000e+00, 3.0000e-3, 1.0000e+00],  # 15 (Poor)
                     [0.0000e+00, 1.0000e-2, 1.0000e+00],  # 16 (Poor)
                     [0.0000e+00, 5.0000e-2, 1.0000e+00],  # 17 (Maybe useful)
                     [0.0000e+00, 1.0000e-1, 1.0000e+00],  # 18 (Recommended)
                     [0.0000e+00, 2.0000e-1, 1.0000e+00],  # 19 (Recommended)
                     [0.0000e+00, 2.5000e-1, 1.0000e+00],  # 20 (Recommended)
                     [0.0000e+00, 3.0000e-1, 1.0000e+00],  # 21 (Recommended)
                     [0.0000e+00, 3.5000e-1, 1.0000e+00],  # 22 (Maybe useful)
                     [0.0000e+00, 4.0000e-1, 1.0000e+00],  # 23 (Maybe useful)
                     # [0.0000e+00, 1.0000e-1, 3.0000e-1, 1.0000e+00]  # 24 (Test) # todo: is it used somewhere? commenting out for now
                     ]

        self.register_buffer("partitions", torch.tensor(partitions, dtype=torch.float))

    def compute_loss(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor], integration: str = 'left', partition=21) -> Dict[str, Tensor]:
        """
        Computes the covariance gradient estimator for the TVO bound.

         TVO = Delta_beto_0 ELBO + sum_{i=1..K-1} Delta_beta_k E_{pi_{beta_k]}}( f(x,z) )], f(x,z) = log p(x,z) - log q(z | x)

         Gradient estimator, eq. 11:
         Nabla E_{pi_{beta_k]}}( f(x,z) )] = E_{pi_{beta_k]}}( Nabla f(x,z) )] + Cov( Nabla log tilde{pi}(z), f(z) )

         In this expression the KLs are concatenated by stochastic layer so the freebits can be applied to each of them.

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i]

        :param partition: selected partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;

        :param num_particles: int; number of samples S in importance sampling of expectations
        :param integration: type of integral approximation (Riemann sum); left, right or trapz
        :return: dictionary with outputs [tvo, elbo, kl, log_f_x,z, N_eff]
        """


        partition = self.partitions[partition]

        num_particles = self.iw

        # compute the effective sample size
        N_eff = self.effective_sample_size(log_px_z, log_pzs, log_qzs)

        def cat_by_layer(log_pzs):
            """sum over the latent dimension (N_i) and concatenate stochastic layers.
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
        if self.freebits is not None:
            kl = self.freebits(kl.unsqueeze(-1))
        kl = batch_reduce(kl)

        # compute log f(x, z) = log p(x, z) - log q(z | x) (ELBO)
        log_f_xz = log_px_z - kl

        # compute log of weights w_s = p(x,z)/q(z|x) (between eq. 13 and 14)
        log_weight = log_f_xz.view(self.bs, num_particles) # log_p_xz - log_q
        heated_log_weight = log_weight.unsqueeze(-1) * partition

        def exponentiate_and_normalize(values, dim=0):
            log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
            return torch.exp(values - log_denominator)

        # compute bar{w^\beta_s} (used in eq. 13)
        heated_normalized_weight = exponentiate_and_normalize(heated_log_weight, dim=1)

        # compute tilde{pi}_beta(z) (eq. 7)
        log_p = log_px_z.view(-1,1) + log_pz # <---- is this correct ?
        log_p = log_p.view(self.bs, num_particles)
        log_qz = log_qz.view(self.bs, num_particles)
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
            torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)), dim=1)

        # compute distances of partitioning
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
        tvo = torch.sum(
        multiplier * (cov_term + torch.sum(w_detached * log_weight.unsqueeze(-1),
        dim=1)), dim=1)

        # multisampling ELBO
        log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
        elbo = torch.mean(log_evidence)

        return {'tvo': tvo, 'elbo': elbo, 'kl': kl, 'log_f_xz': log_f_xz, 'N_eff': N_eff}

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, partition:int = 21, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        # From VariationalInference estimator.
        # Removed '.mean(1)'s and changed namings for TVO

        if self.sequential_computation:
            # warning: here only one `output` will be returned
            output = self._sequential_evaluation(model, x, **kwargs)
        else:
            x_target = self._expand_sample(x)
            output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)

        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        tvo_data = self.compute_loss(log_px_z, log_pz, log_qz, partition=partition)

        # loss
        tvo = tvo_data.get('tvo')
        loss = - tvo

        elbo = tvo_data.get('elbo')

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     'elbo': elbo,
                     'nll': - self._reduce_sample(log_px_z),
                     'kl': self._reduce_sample(tvo_data.get('kl')),
                     'N_eff': tvo_data.get('N_eff')},
            'prior': self.prior_diagnostics(output)
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
