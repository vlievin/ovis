from booster import Diagnostic

from .reinforce import *


class BaseWakeSleep(Reinforce):
    """inspired from https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py"""

    def get_wake_theta_loss(self, iw_data):
        """\Delta_theta = \nabla_theta [L_K]"""
        return - iw_data['L_k']

    def get_sleep_phi_loss(self, model, bs):
        """\Delta_phi = E_{p_{\theta}(z, x)} [ - nabla_phi log q_phi (z | x) ]"""

        # sample data from model: z ~ p(z), x ~ p(x|z)
        with torch.no_grad():
            data = model.sample_from_prior(N=bs * self.mc * self.iw)
            x = data['px'].sample()
            z = data['z']

        # compute log q(z | x)
        qz, meta = model.infer(x, tau=0)
        return - batch_reduce(qz.log_prob(z)), meta

    def get_wake_phi_loss(self, log_qz, iw_data):
        """\Delta_theta = \nabla_theta [L_K]"""
        with torch.no_grad():
            log_wk = iw_data['log_wk']
            v_k = log_wk.softmax(2)
        return - torch.sum(v_k.detach() * log_qz, dim=2)

    def get_loss(self, model, log_qz, iw_data):
        raise NotImplementedError

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[
        Tensor, Diagnostic, Dict]:
        bs = x.size(0)
        output = self.evaluate_model(model, x, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=True)

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1).view(bs, self.mc, self.iw, -1).sum(-1)

        # compute loss
        loss, model_meta = self.get_loss(model, log_qz, iw_data)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                **self.base_loss_diagnostics(**iw_data, **output)
            },
        })

        diagnostics.update(self.additional_diagnostics(output))

        if backward:
            loss.mean().backward()

        output.update(**model_meta)
        return loss, diagnostics, output


class WakeSleep(BaseWakeSleep):
    """Reweighted Wake-Sleep algorithm with *sleep-phase* phi update"""

    def get_loss(self, model, log_qz, iw_data):
        bs = log_qz.size(0)
        phi_loss, meta = self.get_sleep_phi_loss(model, bs)
        theta_loss = self.get_wake_theta_loss(iw_data)
        loss = (theta_loss + phi_loss).mean(1)
        return loss, meta


class WakeWake(BaseWakeSleep):
    """Reweighted Wake-Sleep algorithm with *wake-phase* phi update"""

    def get_loss(self, model, log_qz, iw_data):
        loss = (self.get_wake_theta_loss(iw_data) + self.get_wake_phi_loss(log_qz, iw_data)).mean(1)
        return loss, dict()
