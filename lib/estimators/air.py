from .reinforce import *


class AirReinforce(Reinforce):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.log_1_m_uniform = np.log(1. - 1. / self.iw)

    def forward(self, model: nn.Module,
                x: Tensor,
                backward: bool = False,
                debug: bool = False,
                alpha=1.0,
                beta=1.0,
                mode='copt',
                **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        bs = x.size(0)
        # expand  to match the shape [bs, mc, iw, *dims] and flatten
        x_expanded = self._expand_sample(x)

        # forward pass through the model
        output = model(x_expanded, **kwargs)

        # unpacking data
        log_wk = output.get('elbo_sep')
        log_wk = log_wk.view(bs, self.mc, self.iw)
        # mask = output.get('mask_prev').view(bs, self.mc, self.iw, -1)
        # log_px_z = output.get('data_likelihood').view(bs, self.mc, self.iw)

        # get log q(z^k | x)
        log_qz = output.get('z_pres_likelihood')
        log_qz = log_qz.view(bs, self.mc, self.iw, -1)

        # get baselines and targets
        # baseline_value, baseline_target = [output.get(k) for k in ['baseline_value', 'baseline_target']]
        # baseline_value = baseline_value.view(bs, self.mc, self.iw, -1)
        # baseline_target = baseline_target.view(bs, self.mc, self.iw, -1)
        # baseline_target = baseline_target * mask

        # Compute IW-bound
        # log w_k, w_k = p(x, z^k) / q(z^k | x)
        L_k = torch.logsumexp(log_wk, dim=2) - self.log_iw

        # compute v_k = w_k / \sum_l w_l
        v_k = self.normalized_importance_weights(log_wk)

        # vimco
        v_k_safe = torch.min((1 - 1e-6) * torch.ones_like(v_k), v_k)
        log_gamma_K = self.log_1_m_uniform - torch.log1p(- v_k_safe)

        # define the estimator parameters a,b, alpha
        if mode == 'copt':
            a = b = 1
        elif mode == 'vimco':
            a = 1
            b = 0
        elif mode == 'ww':
            a = b = 0
            alpha = -1
        else:
            raise ValueError(f'Unknown estimator mode `{mode}`.')

        # reinforce
        prefactor_k = a * log_gamma_K - alpha * v_k + b / self.iw
        reinforce_loss = torch.sum(prefactor_k[..., None].detach() * log_qz, dim=(2, 3))

        # compute reinforce loss
        # \nabla L_K = \sum_k (L_K - c_k) \nabla \log q(z^k | x) + \nabla L_K
        # baseline_target = baseline_target - log_px_z[:, :, :, None]
        # prefactors = (baseline_target - baseline_value).detach()
        # prefactors = prefactors * mask
        # prefactors = prefactors.detach()
        # reinforce_loss = (prefactors * log_qz * mask).sum(dim=(2, 3))

        # compute baseline loss
        # baseline_loss = torch.nn.functional.mse_loss(baseline_value, baseline_target.detach(), reduction='none')
        # baseline_loss = baseline_loss * mask
        # baseline_loss = baseline_loss.sum(dim=(3)).mean(dim=2)

        # final loss
        loss = - (L_k + reinforce_loss).mean(1)

        # get diagnostics
        with torch.no_grad():
            kl = output.get('kl')
            nll = output.get('recons')

            # N_eff
            N_eff = torch.exp(2 * torch.logsumexp(log_wk, dim=2) - torch.logsumexp(2 * log_wk, dim=2))
            N_eff = N_eff.mean(1)  # MC

            # print(
            #     f">> Lk: {L_k.mean():.3f}, elbo = {output.get('elbo'):.3f}, kl = {kl:.3f}, inferred_n= {output.get('inferred_n').float().mean():.3f}")

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k.mean(1),
                'nll': nll,
                'kl': kl,
                'r_eff': N_eff / self.iw,
                'ess': N_eff
            },
            'reinforce': {
                'loss': reinforce_loss,
            },
        })

        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
