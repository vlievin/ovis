import logging
import math
import os

import matplotlib.image
import torch
from torchvision.utils import make_grid


@torch.no_grad()
def sample_prior_and_save_img(key, model, logdir, global_step=0, writer=None, N=100, **kwargs):
    """sample the Generative model : z ~ p(z), x ~ p(x|z) and save to .png"""
    x_ = model.sample_from_prior(N, **kwargs).get('px')
    if x_ is None:
        return
    sample = x_.sample()

    # make grid
    nrow = math.floor(math.sqrt(N))
    grid = make_grid(sample, nrow=nrow)

    # normalize
    grid -= grid.min()
    grid /= grid.max()

    # log to tensorboard
    if writer is not None:
        writer.add_image(key, grid, global_step)

    # save the raw image
    img = grid.data.permute(1, 2, 0).cpu().numpy()
    matplotlib.image.imsave(os.path.join(logdir, f"{key}.png"), img)


def get_loggers(logdir, keys=['base', 'train', 'valid', 'test'],
                format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s'):
    """get Logging loggers for a set of `keys`"""
    logging.basicConfig(level=logging.INFO,
                        format=format,
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler(os.path.join(logdir, 'run.log')),
                                  logging.StreamHandler()])

    return (logging.getLogger(k) for k in keys)


def summary2logger(logger, summary, global_step, epoch, best=None, stats_key='loss', exp_id=''):
    """write summary to logging"""
    if not stats_key in summary.keys():
        logger.warning('key ' + str(stats_key) + ' not int output dictionary')
    else:
        message = f"{exp_id:32s}"
        message += f'[{global_step} / {epoch}]   '
        message += ''.join([f'{k} {v:6.2f}   ' for k, v in summary.get(stats_key).items()])
        if 'info' in summary.keys() and 'elapsed-time' in summary['info'].keys():
            message += f'({summary["info"]["elapsed-time"]:.2f}s /iter)'
        if best is not None:
            message += f'   (best: {best[0]:6.2f}  [{best[1]} / {best[2]}])'
        logger.info(message)


def log_summary(summary, global_step, epoch, logger=None, writer=None, **kwargs):
    # add `epoch` to `info/epoch`
    summary['info']['epoch'] = epoch

    # log to both logging and tensorboard
    if logger is not None:
        summary2logger(logger, summary, global_step, epoch, **kwargs)
    if writer is not None:
        summary.log(writer, global_step)


def save_model_and_update_best_elbo(model, eval_summary, global_step, epoch, best_elbo, logdir, key='L_k'):
    elbo = eval_summary['loss'][key]
    prev_elbo, *_ = best_elbo
    if elbo > prev_elbo:
        best_elbo = (elbo, global_step, epoch)
        pth = os.path.join(logdir, "model.pth")
        torch.save(model.state_dict(), pth)

    return best_elbo


def load_model(model, logdir):
    device = next(iter(model.parameters())).device
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth"), map_location=device))


def log_grads_data(analysis_data, base_logger, estimator_id, iw):
    """Log the gradient analysis data to the logger"""
    grad_data = analysis_data.get('grads', {})
    base_logger.info(
        f"{estimator_id}, K = {iw} | snr = {grad_data.get('snr', 0.):.3E}, dsnr = {grad_data.get('dsnr', 0.):.3E}, variance = {grad_data.get('variance', 0.):.3E}, magnitude = {grad_data.get('magnitude', 0.):.3E}")
    snr_data = analysis_data.get('snr', {})
    base_logger.info(
        f"{estimator_id}, K = {iw} | snr | p5 = {snr_data.get('p5', 0.):.3E}, p25 = {snr_data.get('p25', 0.):.3E}, p50 = {snr_data.get('p50', 0.):.3E}, p75 = {snr_data.get('p75', 0.):.3E}, p95 = {snr_data.get('p95', 0.):.3E}")
