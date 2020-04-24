from .reinforce import *


class FactorizedVariationalInference(Reinforce):
    """
    Variational Inference. Using this estimator requires the model to be compatible with the reparametrization trick.

    \nabla_phi \L_K = \E_{p(\eps^{1:K} | x)} [ \sum_k v_k \nabla log w_k ]

    * w_k = p(x,z^k))q(z^k | x)
    * v_k = w_k / \sum_l w_l

    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> \
            Tuple[Tensor, Dict, Dict]:
        bs = x.size(0)
        kwargs.pop('zgrads')  # TODO

        x_target = self._expand_sample(x)
        # use z_grads=False to detach 'z' when computing log p(x|z) but , zkey='z_raw' to evaluate the KL using the z with grads
        zgrads = True
        zkey = 'z'
        detach_qlogits = False
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, zgrads=zgrads, zkey=zkey, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]

        # compute loss differentiable with regards to \theta
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=detach_qlogits)
        L_k, kl, N_eff, log_f_xz = [iw_data[k] for k in ('L_k', 'kl', 'N_eff', 'log_f_xz')]

        # compute v_k
        v_k = self.normalized_importance_weights(log_f_xz)

        # compute control variate c_k
        c_k, meta, _nans = self.compute_control_variate(x, **iw_data, **output)
        #
        with torch.no_grad():
            v_k = v_k - c_k.mean(-1)

        # \nabla_{\phi} L_k = E_{\eps^{1:K}} [ \sum_k v_k \nabla_{\phi} log w_k]
        loss = - (v_k.detach() * log_f_xz).sum(dim=2)  # sum over IW and z

        L_k = L_k.mean(1)
        loss = loss.mean(1)

        output.update(**meta)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k,
                'nll': - self._reduce_sample(log_px_z),
                'kl': self._reduce_sample(kl),
                'r_eff': N_eff / self.iw
            },
            'prior': self._diagnostics(output)
        })

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
