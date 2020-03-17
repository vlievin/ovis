import argparse
import os
import json
import traceback
from shutil import rmtree

import numpy as np
import torch
from booster import Aggregator, Diagnostic
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import VAE, Baseline, ConvVAE
from lib import VariationalInference
from lib import get_datasets
from lib.config import get_config
from lib.gradients import get_gradients_log_total_variance
from lib.logging import sample_model, get_loggers, log_summary, save_model
from lib.utils import notqdm

_sep = os.get_terminal_size().columns * "-"

parser = argparse.ArgumentParser()

# run directory, id and seed
parser.add_argument('--dataset', default='shapes', help='dataset [shapes | binmnist | omniglot | fashion]')
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--data_root', default='data/', help='directory to store the data')
parser.add_argument('--exp', default='sandbox', help='experiment directory')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--seed', default=13, type=int, help='random seed')
parser.add_argument('--rm', action='store_true', help='delete previous run')
parser.add_argument('--silent', action='store_true', help='silence tqdm')
parser.add_argument('--deterministic', action='store_true', help='use deterministic backend')

# epochs, batch size, MC samples, lr
parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--baseline_lr', default=5e-3, type=float, help='learning rate for the weight of the baseline')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--lr_reduce_steps', default=4, type=int, help='number of learning rate reduce steps')

# estimator
parser.add_argument('--estimator', default='reinforce', help='[vi, reinforce, vimco, gs, st-gs]')
parser.add_argument('--mc', default=1, type=int, help='number of Monte-Carlo samples')
parser.add_argument('--iw', default=1, type=int, help='number of Importance-Weighted samples')
parser.add_argument('--iw_valid', default=100, type=int, help='number of Importance-Weighted samples for validation')

# gradients analysis
parser.add_argument('--grad_eval_freq', default=5, type=int, help='frequency for the gradients evaluation')
parser.add_argument('--grad_samples', default=16, type=int, help='number of samples used to evaluate the variance. (at maximum it is size `bs` to avoid using too much memory)')
parser.add_argument('--counterfactuals', default='',
                    help='comma separated list of estimators for which the gradients will be evaluated without being used for optimization.'
                         'example: `reinforce, covbaseline-arithmetic`')

# latent space
parser.add_argument('--prior', default='categorical', help='family of the prior distribution : [categorcial | normal]')
parser.add_argument('--N', default=8, type=int, help='number of latent variables')
parser.add_argument('--K', default=8, type=int, help='number of categories for each latent variable')
parser.add_argument('--kdim', default=0, type=int, help='dimension of the keys for each latent variable')
parser.add_argument('--learn_prior', action='store_true', help='learn the prior')

# model architecture
parser.add_argument('--model', default='vae', help='[vae, conv-vae]')
parser.add_argument('--hdim', default=64, type=int, help='number of hidden units for each layer')
parser.add_argument('--nlayers', default=3, type=int, help='number of hidden layers for the encoder and decoder')
parser.add_argument('--b_nlayers', default=1, type=int, help='number of hidden layers for the baseline')
parser.add_argument('--norm', default='layernorm', type=str, help='normalization layer [none | layernorm | batchnorm]')

opt = parser.parse_args()

if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if opt.silent:
    tqdm = notqdm

# defining the run identifier
counterfactual_estimators = opt.counterfactuals.replace(" ", "").split(",") if len(opt.counterfactuals) else []
use_baseline = any(['-baseline' in e for e in [opt.estimator] + counterfactual_estimators])
run_id = f"{opt.dataset}-{opt.model}-{opt.prior}-{opt.estimator}-seed{opt.seed}"
if len(opt.id) > 0:
    run_id += f"-{opt.id}"
run_id += f"-lr{opt.lr:.1E}-bs{opt.bs}-mc{opt.mc}-iw{opt.iw}+{opt.iw_valid}"
if use_baseline:
    run_id += f"-b{opt.b_nlayers}"
run_id += f"-N{opt.N}-K{opt.K}-kdim{opt.kdim}"
if opt.learn_prior:
    run_id += "-learn-prior"
run_id += f"-arch{opt.hdim}x{opt.nlayers}"
if opt.norm is not 'none':
    run_id += f"-{opt.norm}"
_exp_id = opt.exp

# defining the run directory
logdir = os.path.join(opt.root, opt.exp)
logdir = os.path.join(logdir, run_id)
if os.path.exists(logdir):
    if opt.rm:
        rmtree(logdir)
        os.makedirs(logdir)
else:
    os.makedirs(logdir)

# save configuration
with open(os.path.join(logdir, 'config.json'), 'w') as fp:
    fp.write(json.dumps(vars(opt), default=lambda x: str(x), indent=4))

# wrap the training loop inside a try/except so we can write potential errors to a file.
try:
    # define logger
    base_logger, train_logger, valid_logger = get_loggers(logdir)
    base_logger.info(f"Torch version: {torch.__version__}")

    # setting the random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # get datasets
    dset_train, dset_valid, dset_test = get_datasets(opt)
    x = dset_train[0]
    base_logger.info(f"Dataset size: train = {len(dset_train)}, valid = {len(dset_valid)}")
    base_logger.info(f"Sample: x.shape = {x.shape}, x.min = {x.min():.1f}, x.max = {x.max():.1f}, x.dtype = {x.dtype}")

    # dataloaders
    loader_train = DataLoader(dset_train, batch_size=opt.bs, shuffle=True)
    loader_valid = DataLoader(dset_valid, batch_size=2 * opt.bs, shuffle=True)

    # define model
    torch.manual_seed(opt.seed)
    _MODEL = {'vae': VAE, 'conv-vae': ConvVAE}[opt.model]
    model = _MODEL(x.shape, opt.N, opt.K, opt.hdim, kdim=opt.kdim, nlayers=opt.nlayers, learn_prior=opt.learn_prior, prior=opt.prior, normalization=opt.norm)

    # define baseline
    baseline = Baseline(x.shape, opt.b_nlayers, opt.hdim) if use_baseline else None

    # estimator
    Estimator, config = get_config(opt.estimator)
    estimator = Estimator(baseline=baseline, mc=opt.mc, iw=opt.iw, N=opt.N, K=opt.K, hdim=opt.hdim)

    # valid estimator (it is important that all models are evaluated using the same evaluator)
    config_valid = {'tau': 0, 'zgrads': False}
    estimator_valid = VariationalInference(mc=1, iw=opt.iw_valid, sequential_computation=True)

    # counterfactual estimators:
    # they are use to measure the variance of the gradients given other estimator without using them for optimization
    if len(counterfactual_estimators) :
        c_Estimators, c_configs = zip(*[get_config(c) for c in counterfactual_estimators])
        c_estimators = [Est(baseline=baseline, mc=opt.mc, iw=opt.iw, N=opt.N, K=opt.K, hdim=opt.hdim) for Est in c_Estimators]
    else:
        c_configs = c_estimators = []

    # get device and move models
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    model.to(device)
    estimator.to(device)
    estimator_valid.to(device)

    # optimizers
    optimizers = []
    optimizers += [torch.optim.Adam(model.parameters(), lr=opt.lr)]
    if len(list(estimator.parameters())):
        optimizers += [torch.optim.Adam(estimator.parameters(), lr=opt.baseline_lr)]

    print(f"{_sep}\nModel paramters:")
    for k, v in model.named_parameters():
        print(f"   {k} : {v.numel()}")
    print(f"{_sep}\nEstimator paramters:")
    for k, v in estimator.named_parameters():
        print(f"   {k} : {v.numel()}")
    print(_sep)

    # data aggregator
    agg_train = Aggregator()
    agg_valid = Aggregator()

    # tensorboard writers
    writer_train = SummaryWriter(os.path.join(logdir, 'train'))
    writer_valid = SummaryWriter(os.path.join(logdir, 'valid'))
    counterfactual_writers = [SummaryWriter(os.path.join(logdir, c)) for c in counterfactual_estimators]

    # batch of data for the gradients variance evaluation (at maximum of size bs)
    x_grads_eval = next(iter(loader_train)).to(device)[:opt.grad_samples]

    # run
    best_elbo = (-1e20, 0, 0)
    global_step = 0
    for epoch in range(1, opt.epochs + 1):

        # sample model
        sample_model("prior-sample", model, logdir, global_step=global_step, writer=writer_valid, seed=opt.seed)

        # train epoch
        [o.zero_grad() for o in optimizers]
        model.train()
        agg_train.initialize()
        for x in tqdm(loader_train, desc=_exp_id):
            x = x.to(device)
            loss, diagnostics, output = estimator(model, x, backward=True, **config)
            [o.step() for o in optimizers]
            [o.zero_grad() for o in optimizers]
            agg_train.update(diagnostics)
            global_step += 1
        summary_train = agg_train.data.to('cpu')

        if epoch % opt.grad_eval_freq == 0:
            # estimate the variance of the gradients (for `opt.grad_samples` data points)
            grad_args = {'seed':opt.seed, 'batch_size': opt.bs}
            # current model
            summary_train["loss"]["log_grad_var"], *_ = get_gradients_log_total_variance(estimator, model, x_grads_eval, **grad_args, **config)
            # counter factual estimation of other estimators
            for (c_writer, c_conf, c_est, c) in zip(counterfactual_writers, c_configs, c_estimators, counterfactual_estimators):
                log_grad_var, control_variate_mse = get_gradients_log_total_variance(c_est, model, x_grads_eval, **grad_args, **c_conf)
                summary = Diagnostic({'loss': {'log_grad_var': log_grad_var, 'control_variate_mse': control_variate_mse}})
                summary.log(c_writer, global_step)
                train_logger.info(f" | counterfactual | {c:{max(map(len, counterfactual_estimators))}s} "
                                  f"| log_grad_var = {log_grad_var:.3f}, mse = {summary['loss']['control_variate_mse']:.3f}")

        # valid epoch
        with torch.no_grad():
            model.eval()
            agg_valid.initialize()
            for x in tqdm(loader_train, desc=_exp_id):
                x = x.to(device)
                _, diagnostics, _ = estimator_valid(model, x, backward=False, **config_valid)
                agg_valid.update(diagnostics)
            summary_valid = agg_valid.data.to('cpu')

        # update best elbo and save model
        best_elbo = save_model(model, summary_valid, global_step, epoch, best_elbo, logdir)

        # log to console and tensorboard
        log_summary(summary_train, global_step, epoch, logger=train_logger, writer=writer_train, exp_id=_exp_id)
        log_summary(summary_valid, global_step, epoch, logger=valid_logger, best=best_elbo, writer=writer_valid, exp_id=_exp_id)

        # reduce learning rate
        lr_freq = (opt.epochs // (opt.lr_reduce_steps + 1))
        if epoch % lr_freq == 0:
            for o in optimizers:
                for i, param_group in enumerate(o.param_groups):
                    lr = param_group['lr']
                    new_lr = lr / 2
                    param_group['lr'] = new_lr
                    base_logger.info(f"Reducing lr, group = {i} : {lr:.2E} -> {new_lr:.2E}")


    # write outcome to a file (success, interrupted, error)
    print("## SUCCESS")
    with open(os.path.join(logdir, "success.txt"), 'w') as f:
        f.write(f"Success.")

except KeyboardInterrupt:
    print("## KEYBOARD INTERRUPT")
    with open(os.path.join(logdir, "success.txt"), 'w') as f:
        f.write(f"Failed. Interrupted (keyboard).")

except Exception as ex:
    print("## FAILED. Exception:")
    print("--------------------------------------------------------------------------------")
    traceback.print_exception(type(ex), ex, ex.__traceback__)
    print("--------------------------------------------------------------------------------")
    print("\nException: ", ex, "\n")
    with open(os.path.join(logdir, "success.txt"), 'w') as f:
        f.write(f"Failed. Exception : \n{ex}")
