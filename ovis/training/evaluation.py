import torch
from booster import Diagnostic, Aggregator
from tqdm import tqdm

from ovis.analysis.gradients import get_gradients_statistics
from ovis.training.ops import test_step
from ovis.training.utils import preprocess
from ovis.utils.utils import ManualSeed


def perform_gradients_analysis(opt, global_step, writer_train, train_logger, loader_train, model, estimator, config,
                               exp_id, tqdm=tqdm):
    """
    Perform the gradients analysis for the main estimator and the counterfactual estimators if available
    """
    device = next(iter(model.parameters())).device

    # batch of data for the gradients variance evaluation (at maximum of size bs)
    batch = next(iter(loader_train))
    x, *_ = preprocess(batch, device)
    x = x[:opt.grad_bs]

    # get the configuration for the gradients analysis
    grad_args = {'seed': opt.seed,
                 'batch_size': opt.bs * opt.mc * opt.iw,
                 'mc_samples': opt.grad_samples,
                 'key_filter': opt.grad_key}

    # gradients analysis for the training estimator
    with ManualSeed(seed=opt.seed):
        grad_data, _ = get_gradients_statistics(estimator, model, x, tqdm=tqdm, **grad_args, **config)

    # log the gradient analysis
    with torch.no_grad():
        summary = Diagnostic(grad_data)
        summary.log(writer_train, global_step)
        train_logger.info(f"{exp_id} | grads | "
                          f"snr = {grad_data.get('grads', {}).get('snr', 0.):.3E}, "
                          f"variance = {grad_data.get('grads', {}).get('variance', 0.):.3E}, "
                          f"magnitude = {grad_data.get('grads', {}).get('magnitude', 0.):.3E}, "
                          f"dsnr = {grad_data.get('grads', {}).get('dsnr', -1):.3E}, ")


@torch.no_grad()
def evaluation(model, estimator, config, loader, exp_id, device='cpu', ref_summary=None, max_eval=None, tqdm=tqdm):
    """evaluation epoch to estimate the marginal log-likelihood: L_K \approx log \hat{p(x)}, K -> \infty"""
    k = 0
    model.eval()
    agg = Aggregator()
    for batch in tqdm(loader, desc=f"{exp_id}-eval"):
        x, y = preprocess(batch, device)
        diagnostics = test_step(x, model, estimator, y=y, **config)
        agg.update(diagnostics)
        k += x.size(0)
        if max_eval is not None and k >= max_eval:
            break
    summary = agg.data.to('cpu')

    # compute overfitting L_K^{train} - L_K^{valid}
    if ref_summary is not None:
        summary['loss']['overfitting'] = ref_summary['loss']['L_k'].mean() - summary['loss']['L_k'].mean()

    return summary
