import hashlib

from torch import Tensor
from ..utils.utils import BASE_ARGS_EXCEPTIONS

def get_hash_from_opt(opt, exceptions=None):
    if exceptions is None:
        exceptions = BASE_ARGS_EXCEPTIONS
    filtered_opt_dict = {k: v for k, v in vars(opt).items() if k not in exceptions}
    opt_string = ",".join(("{}={}".format(*i) for i in filtered_opt_dict.items()))
    return hashlib.md5(opt_string.encode('utf-8')).hexdigest()


def get_run_id(opt):
    """define a unique run identifier"""

    deterministic_opt_id = get_hash_from_opt(opt)
    warmup_id = ""
    if opt.alpha > 0:
        warmup_id += f"-{opt.alpha}"
    if opt.warmup > 0:
        warmup_id += "-warmup"
    exp_id = f"{opt.dataset}-{opt.model}-{opt.estimator}-K{opt.iw}-M{opt.mc}{warmup_id}-seed{opt.seed}"
    if opt.id != "":
        exp_id += f"-{opt.id}"
    run_id = f"{exp_id}-{deterministic_opt_id}"
    return run_id, exp_id


def get_number_of_epochs(opt, loader_train):
    """define the number of training epochs based on the lenght of the dataset and the number of steps"""
    epochs = opt.epochs
    iter_per_epoch = len(loader_train.dataset) // opt.bs
    if epochs < 0:
        epochs = 1 + opt.nsteps // iter_per_epoch
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
