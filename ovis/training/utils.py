import hashlib
from typing import *

from torch import Tensor

from ..utils.utils import BASE_ARGS_EXCEPTIONS


def get_hash_from_opt(opt: Dict, exceptions=None):
    if exceptions is None:
        exceptions = BASE_ARGS_EXCEPTIONS
    filtered_opt_dict = {k: v for k, v in opt.items() if k not in exceptions}
    opt_string = ",".join(("{}={}".format(*i) for i in filtered_opt_dict.items()))
    return hashlib.md5(opt_string.encode('utf-8')).hexdigest()


def get_run_id(opt: Dict):
    """define a unique run identifier"""

    hash = get_hash_from_opt(opt)
    warmup_id = ""
    if opt['alpha'] > 0:
        warmup_id += f"-{opt['alpha']}"
    if opt['warmup'] > 0:
        warmup_id += "-warmup"

    base_id = "-".join(f"{opt[k]}" for k in ["dataset", "model", "estimator"])
    base_id += '-' + '-'.join(f'{k}{opt[k]}' for k in ['mc', 'iw', 'seed'])
    id_suffix = f"{opt['id']}-" if opt['id'] != "" else ""
    exp_id = f"{id_suffix}{base_id}-{warmup_id}"
    run_id = f"{exp_id}-{hash}"

    print(">>>>", run_id)

    return run_id, exp_id, hash


def get_number_of_epochs(opt, loader_train):
    """define the number of training epochs based on the lenght of the dataset and the number of steps"""
    epochs = opt['epochs']
    iter_per_epoch = -(-len(loader_train.dataset) // opt['bs'])
    if epochs < 0:
        epochs = 1 + opt['nsteps'] // iter_per_epoch
    return epochs, iter_per_epoch


def preprocess(batch, device):
    """preprocess a batch of data received from the DataLoader"""
    if isinstance(batch, Tensor):
        x = batch.to(device)
        return x, None
    else:
        x, y = batch  # assume receiving a tuple (x,y)
        x = x.to(device)
        y = y.to(device)
        return x, y


def get_dataset_mean(loader_train):
    """Compute the mean over the dataset"""
    _xmean = None
    _n = 0.
    for x in loader_train:
        if not isinstance(x, Tensor):
            x, *_ = x

        k, m = x.size(0), x.sum(0)
        _n += k
        if _xmean is None:
            _xmean = m / k
        else:
            _xmean += (m - k * _xmean) / _n
    return _xmean.unsqueeze(0)


def reduce_lr(optimizers, epoch, epochs, lr_reduce_steps, base_logger):
    lr_freq = (epochs // (lr_reduce_steps + 1))
    if epoch % lr_freq == 0:
        for o in optimizers:
            for i, param_group in enumerate(o.param_groups):
                lr = param_group['lr']
                new_lr = lr / 2
                param_group['lr'] = new_lr
                base_logger.info(f"[Reducing lr] group = {i} : {lr:.2E} -> {new_lr:.2E}")
