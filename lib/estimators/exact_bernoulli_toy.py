from booster import Diagnostic

from .base import *
from .vi import VariationalInference


class ExactGradientBernoulli(VariationalInference):
    def forward(self, model: nn.Module, target: Tensor, backward: bool = False, **kwargs: Any) -> Tuple[
        Tensor, Dict, Dict]:
        """
        Calculates the exact gradient of the Bernoulli toy example.
        :param model: Bernoulli Toy Model
        :param target: target parameter value
        :param backward: perform the backward pass by calling loss.backward()
        :return: exact gradient for the Bernoulli toy example
        """

        logits, output = model.infer(target)

        # expand target as shape [bs * mc * iw]
        target = self._expand_sample(target)

        # E_{b ~ p_theta} [ (b-0.499)**2 ]
        loss = torch.sigmoid(logits)*(1-target)**2 + (1-torch.sigmoid(logits)) * target**2
        loss = loss.mean()

        # prepare diagnostics
        diagnostics = Diagnostic({
            'loss': {'loss': loss,
                     # 'bernoulli_loss': loss,
                     **self._loss_diagnostics()}
        })

        # add diagnostics
        diagnostics.update(self._diagnostics(output))

        if backward:
            loss.mean().backward()

        # forward pass
        output = model(target, **kwargs)

        return loss, diagnostics, output

    def _loss_diagnostics(self):
        tensor = torch.tensor([0.], dtype=torch.float)
        return {'elbo': tensor}

