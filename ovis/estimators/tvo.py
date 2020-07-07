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

    def __init__(self, **kwargs: Any):
        assert not kwargs.get('sequential_computation',
                              False), f"{type(self).__name__} is not Compatible with Sequential Computation"
        super().__init__(**kwargs, reparameterization=False, detach_qlogits=False)

    def get_partition(self, iw, log_beta_min, num_partition, partition_type, partition_id, device):

        if partition_id is not None:
            # map K to a partition accordingly to the figure 3. in the TVO paper
            if partition_id == 'config1':
                # figure 3 in the TVO paper
                if iw < 10:
                    beta_min = 5e-2
                elif iw < 30:
                    beta_min = 2e-1
                else:
                    beta_min = 3e-1
            elif partition_id == 'config2':
                # figure 6 in the TVO paper
                if iw < 10:
                    beta_min = 0.01
                elif iw < 30:
                    beta_min = 0.02
                else:
                    beta_min = 0.03
            else:
                raise ValueError(f"Unknown partition_name = `{partition_id}`")

            log_beta_min = torch.tensor(beta_min, device=device).log10()

        return build_partition(num_partition, partition_type, log_beta_min=log_beta_min, device=device)

    def compute_tvo_loss(self, log_px_z: Tensor,
                         log_pz: Tensor,
                         log_qz: Tensor,
                         log_wk: Tensor = None,
                         integration: str = 'left',
                         num_partition=2,
                         partition_type='log',
                         log_beta_min=-10,
                         partition_id=None,
                         **kwargs: Any) -> Tensor:
        """
        Computes the covariance gradient estimator for the TVO bound.

         TVO = Delta_beta_0 ELBO + sum_{i=1..K-1} Delta_beta_k E_{pi_{beta_k]}}( f(x,z) )], f(x,z) = log p(x,z) - log q(z | x)

         Gradient estimator, eq. 11:
         Nabla E_{pi_{beta_k]}}( f(x,z) )] = E_{pi_{beta_k]}}( Nabla f(x,z) )] + Cov( Nabla log tilde{pi}(z), f(z) )

        :param log_px_z: log p(x | z) of shape [bs, mc, iw]
        :param log_pz: `log p(z) l=1..L]` of shape [bs, mc, iw, L]
        :param log_qz: `log q(z|x) l=1..L]` of shape [bs, mc, iw, L]
        :param log_wk: `log w_k` of shape [bs, mc, iw]
        :param integration: type of integral approximation (Riemann sum); left, right or trapz
        :param num_partition: partition size
        :param log_beta_min: log beta_1 in base 10
        :param partition_id: infer beta_1 based on the the number of particles K [None, config1, config2]
        :return: loss]
        """

        # build the partition
        iw = log_wk.shape[2]
        partition = self.get_partition(iw, log_beta_min, num_partition, partition_type, partition_id, log_px_z.device)

        # log p(x,z)
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
        if iw == 1:
            correction = 1
        else:
            correction = iw / (iw - 1)

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
        return torch.sum(multiplier * (cov_term + torch.sum(w_detached * log_wk.unsqueeze(-1), dim=2)), dim=2).mean()

    def forward(self,
                model: nn.Module,
                x: Tensor,
                backward: bool = False,
                return_diagnostics: bool = True,
                **kwargs: Any) -> Tuple[
        Tensor, Diagnostic, Dict]:

        # update the `config` object with the kwargs
        config = self.get_runtime_config(**kwargs)

        # forward pass, eval of the log probs and iw bound
        log_probs, output = self.evaluate_model(model, x, **config)
        iw_data = self.compute_iw_bound_with_data(**log_probs, **config)

        # compute the TVO loss
        loss = self.compute_tvo_loss(**log_probs, **iw_data, **config)

        # prepare diagnostics
        diagnostics = Diagnostic()
        if return_diagnostics:
            diagnostics = self.base_loss_diagnostics(**iw_data, **log_probs)
            diagnostics.update(self.additional_diagnostics(**output))

        if backward:
            loss.backward()

        return loss, diagnostics, output
