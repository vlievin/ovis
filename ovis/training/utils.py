from torch import Tensor

from run import loader_train


def get_run_id(opt):
    """define a unique identifier given on the parsed config"""
    use_baseline = '-baseline' in opt.estimator
    run_id = f"{opt.dataset}-{opt.optimizer}-{opt.model}-{opt.prior}-{opt.estimator}-seed{opt.seed}"
    if opt.mini:
        run_id = "mini-" + run_id
    if len(opt.id) > 0:
        run_id += f"-{opt.id}"
    run_id += f"-lr{opt.lr:.1E}-bs{opt.bs}-mc{opt.mc}-iw{opt.iw}+{opt.iw_valid}+{opt.iw_test}"
    if use_baseline:
        run_id += f"-b{opt.b_nlayers}"
    run_id += f"-N{opt.N}-K{opt.K}-kdim{opt.kdim}"
    if opt.beta != 1:
        run_id += f"-Beta{opt.beta}"
    if opt.freebits is not None:
        run_id += f"-fb{opt.freebits}"
    if opt.learn_prior:
        run_id += "-learn-prior"
    run_id += f"-arch{opt.hdim}x{opt.nlayers}-L{opt.depth}"
    if opt.skip:
        run_id += "-skip"
    if opt.norm is not 'none':
        run_id += f"-{opt.norm}"
    if opt.dropout > 0:
        run_id += f"-drp{opt.dropout}"
    if opt.model == 'bernoulli_toy':
        run_id += f"-tar{opt.toy_target}"
    if opt.skip:
        run_id += "-skip"
    if opt.warmup > 0:
        run_id += f"-{opt.warmup_mode}-warmup{opt.warmup}-{opt.gamma_min}-{opt.gamma}"
    else:
        if opt.gamma != 1:
            run_id += f"-gamma{opt.gamma}"

    exp_id = f"{opt.exp}-{opt.estimator}-K={opt.iw}"
    if opt.warmup > 0:
        exp_id += f"-{opt.warmup_mode}-wp{opt.warmup}-{opt.gamma_min}-{opt.gamma}"
    elif opt.gamma != 1:
        exp_id += f"-gamma{opt.gamma}"

    return run_id, exp_id, use_baseline


def get_number_of_epochs(opt):
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