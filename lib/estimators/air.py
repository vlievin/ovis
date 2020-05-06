from .reinforce import *


class AirReinforce(VimcoPlus):

    def forward(self, model: nn.Module,
                x: Tensor,
                y: Optional[Tensor] = None,
                backward: bool = False,
                debug: bool = False,
                beta=1.0,
                **kwargs: Any) -> Tuple[Tensor, Dict, Dict]:
        bs = x.size(0)
        # expand  to match the shape [bs, mc, iw, *dims] and flatten
        x_expanded = self._expand_sample(x)
        y = self._expand_sample(y) if y is not None else None

        # forward pass through the model
        output = model(x_expanded, y=y, **kwargs)

        # unpacking data
        log_wk = output.get('elbo_sep')
        log_wk = log_wk.view(bs, self.mc, self.iw)
        # mask = output.get('mask_prev').view(bs, self.mc, self.iw, -1)
        # log_px_z = output.get('data_likelihood').view(bs, self.mc, self.iw)

        # get log q(z^k | x)
        log_qz = output.get('z_pres_likelihood')
        log_qz = log_qz.view(bs, self.mc, self.iw, -1)

        # Compute IW-bound
        # log w_k, w_k = p(x, z^k) / q(z^k | x)
        L_k = torch.logsumexp(log_wk, dim=2) - self.log_iw

        # effective sample size
        ess = self.effective_sample_size(log_wk)

        # compute prefactors
        prefactor_k = self.compute_prefactors(L_k, log_wk, ess, **kwargs)

        # reinforce
        reinforce_loss = torch.sum(prefactor_k[..., None].detach() * log_qz, dim=(2, 3))

        # final loss
        loss = - (L_k + reinforce_loss).mean(1)

        # get diagnostics
        with torch.no_grad():
            kl = output.get('kl')
            nll = output.get('recons')

            # compute accuracy weighted by normalized weights v_k
            if y is not None:
                v_k = log_wk.softmax(2)
                inferred_n = output.get('inferred_n')
                correct = (inferred_n == y).float().view(-1, self.mc, self.iw)
                weighted_correct = (v_k * correct).sum(2)
                accuracy = weighted_correct.mean(1)
            else:
                accuracy = None

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {
                'loss': loss,
                'elbo': L_k.mean(1),
                'nll': nll,
                'kl': kl,
                'accuracy': accuracy,
                'r_eff': ess.mean(1) / self.iw,
                'ess': ess.mean(1),
            },
            'reinforce': {
                'loss': reinforce_loss,
            },
        })

        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        return loss, diagnostics, output
