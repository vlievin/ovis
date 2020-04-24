from .reinforce import *


class BaseWakeSleep(Reinforce):
    """inspired from https://github.com/vmasrani/tvo/blob/master/discrete_vae/losses.py"""

    def get_sleep_phi_loss(self, model, bs):
        """\Delta_phi = E_{p_{\theta}(z | x)} [ - nabla_phi log q_phi (z | x) ]"""

        # sample data from model: z ~ p(z), x ~ p(x|z)
        with torch.no_grad():
            data = model.sample_from_prior(N=bs)
            x = data['px'].sample()
            z = data['z']

        # compute log q(z | x)
        qz, qlogits = model.infer(x, tau=0, mc=1, iw=1)
        return - batch_reduce(qz.log_prob(z)), qlogits

    def get_wake_theta_loss(self, iw_data):
        """\Delta_theta = \nabla_theta [L_K]"""
        return - iw_data['L_k']

    def get_wake_phi_loss(self, log_qz, iw_data):
        """\Delta_theta = \nabla_theta [L_K]"""
        with torch.no_grad():
            log_wk = iw_data['log_wk']
            v_k = self.normalized_importance_weights(log_wk)
        return - torch.sum(v_k.detach() * log_qz, dim=2)

    def get_loss(self, model, log_qz, iw_data):
        raise NotImplementedError

    def forward(self, model: nn.Module, x: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        bs = x.size(0)
        x_target = self._expand_sample(x)
        output = self.evaluate_model(model, x, x_target, mc=self.mc, iw=self.iw, **kwargs)
        log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
        iw_data = self.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=True)
        L_k, kl, N_eff, log_wk = [iw_data[k] for k in ('L_k', 'kl', 'N_eff', 'log_wk')]

        # concatenate all q(z_l| *, x)
        log_qz = torch.cat(log_qz, 1).view(bs, self.mc, self.iw, -1).sum(-1)

        # compute loss
        loss, aux_output = self.get_loss(model, log_qz, iw_data)
        output.update(**aux_output)

        # MC averaging
        L_k = L_k.mean(1)
        loss = loss.mean(1)

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k,
                'nll': - self._reduce_sample(log_px_z),
                'kl': self._reduce_sample(kl),
                'r_eff': N_eff / self.iw
            },
        })

        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output


class WakeSleep(BaseWakeSleep):
    def get_loss(self, model, log_qz, iw_data):
        bs = log_qz.size(0)
        phi_loss, qlogits = self.get_sleep_phi_loss(model, bs)
        theta_loss = self.get_wake_theta_loss(iw_data)
        return theta_loss + phi_loss, {'qlogits' : [qlogits]}


class WakeWake(BaseWakeSleep):

    def get_loss(self, model, log_qz, iw_data):
        return self.get_wake_theta_loss(iw_data) + self.get_wake_phi_loss(log_qz, iw_data), dict()
