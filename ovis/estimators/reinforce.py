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
        assert not kwargs.get('sequential_computation',
                              False), f"{type(self).__name__} is not Compatible with Sequential Computation"
        super().__init__(**kwargs, reparameterization=False, detach_qlogits=True)
        self.baseline = baseline
        control_loss_weight = 1. if baseline is not None else 0.
        self.register_buffer('control_loss_weight', torch.tensor(control_loss_weight))

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

    def forward(self,
                model: nn.Module,
                x: Tensor,
                backward: bool = False,
                return_diagnostics: bool = True,
                **kwargs: Any) -> Tuple[Tensor, Diagnostic, Dict]:

        # update the `config` object with the `kwargs`
        config = self.get_runtime_config(**kwargs)

        # forward pass and eval of the log probs
        log_probs, output = self.evaluate_model(model, x, **config)
        iw_data = self.compute_iw_bound_with_data(**log_probs, **config)

        # unpack data
        L_k, log_wk = [iw_data[k] for k in ('L_k', 'log_wk')]
        log_qz = log_probs['log_qz']

        # compute the score function d_k = \log Z - v_k
        v_k = log_wk.softmax(2)
        d_k = L_k[:, :, None] - v_k

        # compute the control variate c_k
        c_k = self.compute_control_variate(x, d_k=d_k, v_k=v_k, **log_probs, **iw_data, **config)

        # compute the loss for the inference network (sum over `iw` and `log q(z|x) groups`, avg. over `mc`)
        loss_phi = - ((d_k - c_k).detach() * log_qz.sum(3)).sum(2).mean(1)

        # compute the loss for the generative model
        loss_theta = - L_k.mean(1)

        # compute control variate and MSE
        control_variate_loss = self.compute_control_variate_loss(d_k, c_k)

        # final loss
        loss = loss_theta + loss_phi + self.control_loss_weight * control_variate_loss
        loss = loss.mean()

        # prepare diagnostics
        diagnostics = Diagnostic()
        if return_diagnostics:
            diagnostics = self.base_loss_diagnostics(**iw_data, **log_probs)
            diagnostics.update(self.additional_diagnostics(**output))
            diagnostics.update({'reinforce': {'mse': control_variate_loss}})

        if backward:
            loss.backward()

        return loss, diagnostics, output
