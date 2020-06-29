from booster.utils import logging_sep

from ovis.estimators.config import get_config


def evaluate_and_log(estimator, model, x, config, base_logger, desc):
    """Evaluate the model using the estimator on a mini-batch of data and log to the Logger"""
    print(logging_sep())
    loss, diagnostics, output = estimator(model, x, backward=False, **config)
    base_logger.info(
        f"{desc} | L_{estimator.iw} = {diagnostics['loss']['elbo'].mean().item():.6f}, "
        f"KL = {diagnostics['loss']['kl'].mean().item():.6f}, "
        f"NLL = {diagnostics['loss']['nll'].mean().item():.6f}, "
        f"ESS = {diagnostics['loss']['ess'].mean().item():.6f}")

    return diagnostics


def init_estimator(estimator_id, iw):
    """initialize the gradient estimator based on the `estimator_id` and the number of particles `iw`"""
    Estimator, config = get_config(estimator_id)
    return Estimator(baseline=None, mc=1, iw=iw, **config), config


def log_grads_data(analysis_data, base_logger, estimator_id, iw):
    """Log the gradient analysis data to the logger"""
    grad_data = analysis_data.get('grads', {})
    base_logger.info(
        f"{estimator_id}, K = {iw} | snr = {grad_data.get('snr', 0.):.3E}, dsnr = {grad_data.get('dsnr', 0.):.3E}, variance = {grad_data.get('variance', 0.):.3E}, magnitude = {grad_data.get('magnitude', 0.):.3E}")
    snr_data = analysis_data.get('snr', {})
    base_logger.info(
        f"{estimator_id}, K = {iw} | snr | p5 = {snr_data.get('p5', 0.):.3E}, p25 = {snr_data.get('p25', 0.):.3E}, p50 = {snr_data.get('p50', 0.):.3E}, p75 = {snr_data.get('p75', 0.):.3E}, p95 = {snr_data.get('p95', 0.):.3E}")
