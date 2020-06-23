import logging
import math
import os
import sys
from shutil import rmtree

import matplotlib.image
import torch
from torchvision.utils import make_grid


@torch.no_grad()
def sample_model(key, model, logdir, global_step=0, writer=None, N=100, seed=None, **kwargs):
    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    # sample model
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

    if seed is not None:
        # set a new random seed
        torch.manual_seed(_seed)


def get_loggers(logdir, keys=['base', 'train', 'valid', 'test'], format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s'):
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
        pth = os.path.join(logdir, "model.pt")
        torch.save(model.state_dict(), pth)

    return best_elbo


def load_model(model, logdir):
    device = next(iter(model.parameters())).device
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pt"), map_location=device))


def init_logging_directory(opt, run_id):
    """initialize the directory where will be saved the config, model's parameters and tensorboard logs"""
    logdir = os.path.join(opt.root, opt.exp)
    logdir = os.path.join(logdir, run_id)
    if os.path.exists(logdir):
        if opt.rm:
            rmtree(logdir)
            os.makedirs(logdir)
    else:
        os.makedirs(logdir)

    return logdir