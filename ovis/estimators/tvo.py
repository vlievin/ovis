from booster import Diagnostic

from .vi import *


def build_partition(num_partitions, partition_type, log_beta_min=-10,
                    device=None):
    """Create a non-decreasing sequence of values between zero and one.
    See https://en.wikipedia.org/wiki/Partition_of_an_interval.
    Args:
        num_partitions: length of sequence minus one
        partition_type: \'linear\' or \'log\'
        log_beta_min: log (base ten) of beta_min. only used if partition_type
            is log. default -10 (i.e. beta_min = 1e-10).
        device: torch.device object (cpu by default)
    Returns: tensor of shape [num_partitions + 1]
    """
    if device is None:
        device = torch.device('cpu')
    if num_partitions == 1:
        partition = torch.tensor([0, 1], dtype=torch.float, device=device)
    else:
        if partition_type == 'linear':
            partition = torch.linspace(0, 1, steps=num_partitions + 1,
                                       device=device)
        elif partition_type == 'log':
            partition = torch.zeros(num_partitions + 1, device=device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(
                log_beta_min, 0, steps=num_partitions, device=device,
                dtype=torch.float)
        else:
            raise ValueError(f"Unknown TVO partition type `{partition_type}`")
    return partition


class ThermoVariationalObjective(VariationalInference):
    """
    The Thermodynamic Variational Inference https://arxiv.org/pdf/1907.00031.pdf / https://github.com/vmasrani/tvo
    """

    def get_partition(self, log_beta_min, num_partition, partition_type, partition_id, device):

        if partition_id is not None:
            # map K to a partition accordingly to the figure 3. in the TVO paper
            if partition_id == 'config1':
                # figure 3 in the TVO paper
                if self.iw < 10:
                    beta_min = 5e-2
                elif self.iw < 30:
                    beta_min = 2e-1
                else:
                    beta_min = 3e-1
            elif partition_id == 'config2':
                # figure 6 in the TVO paper
                if self.iw < 10:
                    beta_min = 0.01
                elif self.iw < 30:
                    beta_min = 0.02
                else:
                    beta_min = 0.03
            else:
                raise ValueError(f"Unknown partition_name = `{partition_id}`")

            log_beta_min = torch.tensor(beta_min, device=device).log10()

        return build_partition(num_partition, partition_type, log_beta_min=log_beta_min, device=device)

    def compute_loss(self, log_px_z: Tensor, log_pzs: List[Tensor], log_qzs: List[Tensor], integration: str = 'left',
                     num_partition=2, partition_type='log', log_beta_min=-10, partition_id=None, **kwargs: Any) -> \
            Dict[str, Tensor]:
        """
        Computes the covariance gradient estimator for the TVO bound.

         TVO = Delta_beta_0 ELBO + sum_{i=1..K-1} Delta_beta_k E_{pi_{beta_k]}}( f(x,z) )], f(x,z) = log p(x,z) - log q(z | x)

         Gradient estimator, eq. 11:
         Nabla E_{pi_{beta_k]}}( f(x,z) )] = E_{pi_{beta_k]}}( Nabla f(x,z) )] + Cov( Nabla log tilde{pi}(z), f(z) )

        :param log_px_z: log p(x | z) of shape [bs * mc * iw]
        :param log_pzs: [log p(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i] (one value per stochastic layer)
        :param log_qzs: [log q(z_i | *) l=1..L], each of of shape [bs * mc * iw, N_i] (one value per stochastic layer)
        :param integration: type of integral approximation (Riemann sum); left, right or trapz
        :param num_partition: partition size
        :param log_beta_min: log beta_1 in base 10
        :param partition_id: infer beta_1 based on the the number of particles K [None, config1, config2]
        :return: dictionary with keys [tvo, *iw_data]
        """

        partition = self.get_partition(log_beta_min, num_partition, partition_type, partition_id, log_px_z.device)
        iw_data = self.compute_iw_bound(log_px_z, log_pzs, log_qzs, detach_qlogits=False,
                                        request=['log_px_z', 'log_pz', 'log_qz'], **kwargs)
        log_wk, log_px_z, log_pz, log_qz = [iw_data[k] for k in ['log_wk', 'log_px_z', 'log_pz', 'log_qz']]
        log_p = log_px_z + log_pz

        # compute log of weights w_s = p(x,z)/q(z|x) (between eq. 13 and 14)
        heated_log_wk = log_wk[..., None] * partition[None, None, None, :]

        # compute bar{w^\beta_s} (used in eq. 13)
        heated_normalized_weight = heated_log_wk.softmax(2)

        # compute tilde{pi}_beta(z) (eq. 7)
        thermo_logp = partition * log_p.unsqueeze(-1) + (1 - partition) * log_qz.unsqueeze(-1)

        # compute product in eq. 13 with f(x,z) = log[p(x,z)/q(z|x)]
        wf = heated_normalized_weight * log_wk.unsqueeze(-1)
        w_detached = heated_normalized_weight.detach()

        # num_particles S for importance sampling as
        # described in subsection Expectations of Section 4
        if self.iw == 1:
            correction = 1
        else:
            correction = self.iw / (self.iw - 1)

        # compute covariance (eq. 12)
        # .detach() makes sure PyTorch does not differentiate f_lambda(z) term.
        cov_term = correction * torch.sum(
            w_detached * (log_wk.unsqueeze(-1) - torch.sum(wf, dim=2, keepdim=True)).detach()
            * (thermo_logp - torch.sum(thermo_logp * w_detached, dim=2, keepdim=True)), dim=2)

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
        tvo = torch.sum(multiplier * (cov_term + torch.sum(w_detached * log_wk.unsqueeze(-1), dim=2)), dim=2)

        for k in ['log_px_z', 'log_pz', 'log_qz']:
            iw_data.pop(k)

        return {'tvo': tvo, **iw_data}

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, return_diagnostics: bool = True,
                **kwargs: Any) -> Tuple[
        Tensor, Diagnostic, Dict]:
        # From VariationalInference estimator.
        # Removed '.mean(1)'s and changed namings for TVO

        if self.sequential_computation:
            # warning: here only one `output` will be returned
            output = self.sequential_evaluation(model, x, **kwargs)
        else:
            output = self.evaluate_model(model, x, **kwargs)

        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        tvo_data = self.compute_loss(log_px_z, log_pz, log_qz, **kwargs)

        # loss + L_K
        loss = - tvo_data.get('tvo').mean(1)

        # prepare diagnostics
        diagnostics = Diagnostic()

        if return_diagnostics:
            diagnostics.update({
                'loss': {'loss': loss,
                         **self.base_loss_diagnostics(**tvo_data, **output)}
            })

            diagnostics.update(self.additional_diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
