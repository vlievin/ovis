from .vi import *
from .vimco import Vimco


class BaseWakeSleep(Vimco):
    """inspired from https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py"""

    def get_wake_theta_loss(self, L_k: Tensor = None, **kwargs):
        """\Delta_theta = \nabla_theta [L_K]"""
        return - L_k.mean()

    def get_sleep_phi_loss(self, model: nn.Module, bs: int = 1, mc: int = 1, iw: int = 1, **kwargs):
        """
        \Delta_phi = E_{p_{\theta}(z, x)} [ - nabla_phi log q_phi (z | x) ]
        **warnings**: only implemented for the simple VAE model
        """

        # sample data from model: z ~ p(z), x ~ p(x|z)
        with torch.no_grad():
            data = model.sample_from_prior(N=bs * mc * iw)
            x = data['px'].sample()
            z = data['z']

        # compute log q(z | x)
        qz = model.infer(x, tau=0)
        logqz = batch_reduce(qz.log_prob(z))
        return - logqz.mean()

    def get_wake_phi_loss(self, log_qz: Tensor, log_wk: Tensor, **kwargs):
        """\Delta_theta = \nabla_theta [L_K]"""
        with torch.no_grad():
            v_k = log_wk.softmax(2)
        return - torch.sum(v_k.detach() * log_qz.sum(3), dim=2).mean()

    def get_loss(self, model, **kwargs):
        raise NotImplementedError

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, return_diagnostics: bool = True,
                **kwargs: Any) -> Tuple[Tensor, Diagnostic, Dict]:
        # update the `config` object with the kwargs
        config = self.get_runtime_config(**kwargs)

        # forward pass and eval of the log probs
        log_probs, output = self.evaluate_model(model, x, **config)
        iw_data = self.compute_log_weights_and_iw_bound(**log_probs, **config)

        # compute loss
        loss = self.get_loss(model, **log_probs, **iw_data, **output)

        # prepare diagnostics
        diagnostics = Diagnostic()
        if return_diagnostics:
            diagnostics = self.base_loss_diagnostics(**iw_data, **log_probs)
            diagnostics.update(self.additional_diagnostics(**output))

        if backward:
            loss.backward()

        return loss, diagnostics, output


class WakeSleep(BaseWakeSleep):
    """Reweighted Wake-Sleep algorithm with *sleep-phase* phi update"""

    def get_loss(self, model, **kwargs):
        bs = kwargs['log_qz'].size(0)
        phi_loss = self.get_sleep_phi_loss(model, bs=bs, **kwargs)
        theta_loss = self.get_wake_theta_loss(**kwargs)
        return theta_loss + phi_loss


class WakeWake(BaseWakeSleep):
    """Reweighted Wake-Sleep algorithm with *wake-phase* phi update"""

    def get_loss(self, model, **kwargs):
        theta_loss = self.get_wake_theta_loss(**kwargs)
        phi_loss = self.get_wake_phi_loss(**kwargs)
        return theta_loss + phi_loss
