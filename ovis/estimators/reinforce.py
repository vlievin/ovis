from booster import Diagnostic

from .vi import *


class Reinforce(VariationalInference):
    """
   The "Reinforce" or "score-function" gradient estimator. The gradients of the generative model are given by
      * d L_K/ d\theta = \E_q(z_1, ... z_K | x) [ d/d\theta \log Z ]

    The gradients of the inference network are given by:
      * d L_K/ d\phi = \E_q(z_1, ... z_K | x) [ \sum_k d_k h_k ]

    Where
      * d_k = \log Z - v_k
      * v_k = w_k / \sum_l w_l
      * h_k = d/d\phi \log q(z_k | x)

    Using a control variate c_k, the gradient estimator g of the parameters of the inference network is:
      * g = \sum_k (d_k - c_k) h_k

    The control variate is a neural baseline `c(x)` if a model `baseline` is provided, otherwise `c(x)=0`

    """

    def __init__(self, baseline: Optional[nn.Module] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.detach_qlogits = True  # makes `L_k` differentiable only w.r.t `\theta` (givens no reparameterization)
        self.control_loss_weight = 1. if baseline is not None else 0.
        assert self.sequential_computation == False

    def compute_control_variate(self, x: Tensor, **data: Dict[str, Any]) -> Union[float, Tensor]:
        """
        Compute the baseline (a.k.a control variate). Use the Neural Baseline model if available else the baseline is zero.
        :param x: observation
        :param data: additional data
        :return: control variate of shape [bs, mc, iw]
        """

        if self.baseline is None:
            return 0.

        return self.baseline(x).view((x.size(0), 1, 1))

    def compute_control_variate_loss(self, d_k, c_k):
        """MSE between the score function and the control variate"""
        return (c_k - d_k.detach()).pow(2).mean(dim=(1, 2))

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, debug: bool = False,
                return_diagnostics: bool = True, **kwargs: Any) -> Tuple[
        Tensor, Diagnostic, Dict]:

        bs = x.size(0)
        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        output.update(**self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=self.detach_qlogits, **kwargs))

        # unpack data
        L_k, kl, log_wk = [output[k] for k in ('L_k', 'kl', 'log_wk')]

        # reshape `log_qz` and exclude the auxiliary samples
        log_qz = torch.cat([l.view(l.size(0), -1) for l in log_qz], 1)
        log_qz = log_qz.view(bs, self.mc, self.iw + self.auxiliary_samples, -1)[:, :, :self.iw]

        # compute the score function d_k = \log Z - v_k
        v_k = log_wk.softmax(2)
        d_k = L_k[:, :, None] - v_k

        # compute the control variate c_k
        c_k = self.compute_control_variate(x, d_k=d_k, v_k=v_k, **output, **kwargs)

        # compute the loss for the inference network
        loss_phi = - ((d_k - c_k).unsqueeze(-1).detach() * log_qz).sum(dim=(2, 3)).mean(1)

        # compute the loss for the generative model
        loss_theta = - L_k.mean(1)

        # compute control variate and MSE
        control_variate_loss = self.compute_control_variate_loss(d_k, c_k)

        # final loss
        loss = loss_theta + loss_phi + self.control_loss_weight * control_variate_loss

        # prepare diagnostics
        diagnostics = Diagnostic()

        if return_diagnostics:
            diagnostics.update({
                'loss': {
                    'loss': loss,
                    **self.base_loss_diagnostics(**output)
                },
                'reinforce': {
                    'mse': control_variate_loss
                }
            })

            diagnostics.update(self.additional_diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
