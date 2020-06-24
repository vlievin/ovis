import sys

from torch import Tensor


def get_run_id(opt):
    """define a unique identifier given on the parsed config"""
    exp_id = "-".join(sys.argv).replace("--", "").replace("run.py-", "")
    use_baseline = '-baseline' in opt.estimator
    run_id = f"{opt.dataset}-{opt.model}-{opt.estimator}-seed{opt.seed}-{exp_id}"
    return run_id, exp_id, use_baseline


def get_number_of_epochs(opt, loader_train):
    """define the number of training epochs based on the lenght of the dataset and the number of steps"""
    epochs = opt.epochs
    iter_per_epoch = len(loader_train.dataset) // opt.bs
    if epochs < 0:
        epochs = 1 + opt.nsteps // iter_per_epoch
    return epochs, iter_per_epoch


def preprocess(batch, device):
    if isinstance(batch, Tensor):
        x = batch.to(device)
        return x, None
    else:
        x, y = batch  # assume receiving a tuple (x,y)
        x = x.to(device)
        y = y.to(device)
        return x, y


def get_dataset_mean(loader_train):
    """Compute the mean over the dataset, this is used to initialize SBMs"""
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
