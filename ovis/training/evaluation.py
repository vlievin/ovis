import torch
from booster import Diagnostic, Aggregator
from booster.utils import logging_sep
from tqdm import tqdm

from ovis.analysis.gradients import get_gradients_statistics
from ovis.training.ops import test_step
from ovis.training.utils import preprocess
from ovis.utils.utils import ManualSeed


def analyse_gradients_and_log(opt, global_step, writer_train, train_logger, loader_train, model, estimator, config,
                              exp_id, tqdm=tqdm):
    """
    Analyse of the gradients (SNR, variance, DSNR) and log to Tensorboard and the Logger
    :param opt: parsed args
    :param global_step: training step
    :param writer_train: Tensorboard writer
    :param train_logger: Logging logger
    :param loader_train: pytorch dataloader
    :param model: nn.Module
    :param estimator: gradient estimator
    :param config: configuration dictionary for the gradient estimator
    :param exp_id: experiment identifier
    :param tqdm: tqdm callable (for customization)
    :return: None
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
        grad_data.log(writer_train, global_step) # Tensorboard logging
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

    # get the Diagnostic object
    summary = agg.data.to('cpu')

    # estimate overfitting L_K^{train} - L_K^{valid}
    if ref_summary is not None:
        summary['loss']['overfitting'] = ref_summary['loss']['L_k'].mean() - summary['loss']['L_k'].mean()

    return summary


def evaluate_minibatch_and_log(estimator, model, x, config, base_logger, desc):
    """Evaluate the model using the estimator on a mini-batch of data and log to the Logger"""
    print(logging_sep())
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    base_logger.info(
        f"{desc} | L_{estimator.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, "
        f"KL = {diagnostics['loss']['kl'].mean().item():.6f}, "
        f"NLL = {diagnostics['loss']['nll'].mean().item():.6f}, "
        f"ESS = {diagnostics['loss']['ess'].mean().item():.6f}")
    return diagnostics