import argparse
import json
import os
import sys
import socket
import traceback
from copy import copy
from shutil import rmtree

import numpy as np
import torch
from booster import Aggregator, Diagnostic
from torch import Tensor
from torch.optim import Adam, Adamax, SGD, RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import get_datasets
from lib.analysis import total_derivatives_analysis, latent_activations
from lib.estimators import VariationalInference, AirReinforce
from lib.estimators.config import get_config
from lib.gradients import get_gradients_statistics
from lib.logging import sample_model, get_loggers, log_summary, save_model_and_update_best_elbo, load_model
from lib.models import VAE, Baseline, ConvVAE, GaussianToyVAE, GaussianMixture, BernoulliToyModel, HierarchicalVae, \
    SigmoidBeliefNetwork, AIR
from lib.ops import training_step, test_step
from lib.utils import notqdm, LinearSchedule


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    # run directory, id and seed
    parser.add_argument('--dataset', default='shapes',
                        help='dataset [shapes | binmnist | omniglot | fashion | gmm-toy | gaussian-toy | bernoulli-toy]')
    parser.add_argument('--mini', action='store_true', help='use a sub-sampled version of the dataset')
    parser.add_argument('--root', default='runs/', help='directory to store training logs')
    parser.add_argument('--data_root', default='data/', help='directory to store the data')
    parser.add_argument('--exp', default='sandbox', help='experiment directory')
    parser.add_argument('--id', default='', type=str, help='run id suffix')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--workers', default=1, type=int, help='dataloader workers')
    parser.add_argument('--rm', action='store_true', help='delete previous run')
    parser.add_argument('--silent', action='store_true', help='silence tqdm')
    parser.add_argument('--deterministic', action='store_true', help='use deterministic backend')
    parser.add_argument('--sequential_computation', action='store_true',
                        help='compute each iw sample sequential during validation')
    parser.add_argument('--test_sequential_computation', action='store_true',
                        help='compute each iw sample sequential during validation')

    # epochs, batch size, MC samples, lr
    parser.add_argument('--epochs', default=-1, type=int, help='number of epochs (use n_steps if `epochs` < 0)')
    parser.add_argument('--nsteps', default=500000, type=int, help='number of optimization steps')
    parser.add_argument('--optimizer', default='adam', help='[sgd | adam | adamax | rmsprop]')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--freebits', default=None, type=float, help='freebits per layer')
    parser.add_argument('--baseline_lr', default=5e-3, type=float, help='learning rate for the weight of the baseline')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--valid_bs', default=50, type=int, help='evaluation batch size')
    parser.add_argument('--test_bs', default=50, type=int, help='evaluation batch size')
    parser.add_argument('--grad_bs', default=10, type=int, help='grads evaluation batch size')
    parser.add_argument('--lr_reduce_steps', default=0, type=int, help='number of learning rate reduce steps')
    parser.add_argument('--only_train_set', action='store_true',
                        help='only use the training dataset: useful to study optimization behaviour.')

    # estimator
    parser.add_argument('--estimator', default='reinforce', help='[vi, reinforce, vimco, gs, st-gs]')
    parser.add_argument('--mc', default=1, type=int, help='number of Monte-Carlo samples')
    parser.add_argument('--iw', default=1, type=int, help='number of Importance-Weighted samples')
    parser.add_argument('--beta', default=1.0, type=float, help='Beta weight for the KL term (i.e. Beta-VAE)')
    parser.add_argument('--gamma', default=1.0, type=float, help='Gamma weight for the unormalized weights: log w_k^gamma')
    parser.add_argument('--warmup', default=0, type=int, help='period of the posterior warmup (Gamma : 0 -> 1)')
    parser.add_argument('--warmup_offset', default=0, type=int, help='number of steps before increasing beta')
    parser.add_argument('--warmup_mode', default='log', type=str, help='interpolation mode [linear | log]')
    parser.add_argument('--gamma_min', default=1e-2, type=float, help='minimum gamma value')
    parser.add_argument('--warmup_estimator', default='', help='estimator to use during the warm-up')

    # evaluation
    parser.add_argument('--eval_freq', default=10, type=int, help='frequency for the evaluation [test set + grads]')
    parser.add_argument('--max_eval', default=None, type=int, help='maximum number of data points for evaluation')
    parser.add_argument('--iw_valid', default=100, type=int,
                        help='number of Importance-Weighted samples for validation')
    parser.add_argument('--iw_test', default=1000, type=int, help='number of Importance-Weighted samples for testing')

    # total derivatives analysis
    parser.add_argument('--mc_analysis', default=0, type=int,
                        help='number of Monte-Carlo samples for the total derivatives analysis')

    # active units analysis
    parser.add_argument('--mc_au_analysis', default=0, type=int,
                        help='number of Monte-Carlo samples used to estimate the number of active units')
    parser.add_argument('--npoints_au_analysis', default=1000, type=int,
                        help='number of data points')

    # gradients analysis
    parser.add_argument('--grad_samples', default=100, type=int,
                        help='number of samples used to evaluate the variance.')
    parser.add_argument('--grad_key', default='tensor:qlogits', type=str,
                        help='identifiant of the parameters/tensor for the gradients analysis')
    parser.add_argument('--individual_grads', action='store_true',
                        help='Compute expected gradients for single datapoints instead of over the mini-batch.')
    parser.add_argument('--counterfactuals', default='',
                        help='comma separated list of estimators for which the gradients will be evaluated without being used for optimization.'
                             'example: `reinforce, covbaseline-arithmetic`')
    parser.add_argument('--counterfactuals_iw', default='',
                        help='comma separated list of number of samples.'
                             'example: `8,16,32`')
    parser.add_argument("--oracle", default="", type=str,
                        help="oracle estimator to find the `true` gradients direction (estimator id)")
    parser.add_argument("--oracle_iw_samples", default=32, type=int, help="IW samples")
    parser.add_argument("--oracle_mc_samples", default=1000, type=int, help="MC samples")

    # latent space
    parser.add_argument('--prior', default='categorical',
                        help='family of the prior distribution : [categorcial | normal]')
    parser.add_argument('--N', default=8, type=int, help='number of latent variables')
    parser.add_argument('--K', default=8, type=int, help='number of categories for each latent variable')
    parser.add_argument('--kdim', default=0, type=int, help='dimension of the keys for each latent variable')
    parser.add_argument('--learn_prior', action='store_true', help='learn the prior')

    # model architecture
    parser.add_argument('--model', default='vae', help='[vae, conv-vae, hierarchical, bernoulli_toy]')
    parser.add_argument('--skip', action='store_true', help='use skip connections')
    parser.add_argument('--hdim', default=64, type=int, help='number of hidden units for each layer')
    parser.add_argument('--nlayers', default=3, type=int, help='number of hidden layers for the encoder and decoder')
    parser.add_argument('--depth', default=3, type=int,
                        help='number of stochastic layers when using hierarchical models')
    parser.add_argument('--b_nlayers', default=1, type=int, help='number of hidden layers for the baseline')
    parser.add_argument('--norm', default='none', type=str,
                        help='normalization layer [none | layernorm | batchnorm]')
    parser.add_argument('--dropout', default=0, type=float, help='dropout value')
    parser.add_argument('--toy_target', default=0.499, type=float, help='target in Bernoulli toy example')

    opt = parser.parse_args()

    return opt


def define_counterfactuals(opt):
    """
    Counterfactual estimators are estimators that are only used for evaluation.
    They are used to evaluate the gradients of another estimator (counterfactual) given the parameters obtained
    following the optimization trajectory of the main estimator. Counterfactual estimators answer the question:
    "What if we had used this estimator at that point of the parameter space?"

    :param opt: parsed args
    :return: ounterfactual estimator instances, samples for each estimator, ids
    """
    if len(opt.counterfactuals):
        counterfactual_estimators_ids = opt.counterfactuals.replace(" ", "").split(",")

        # construct product with the number of samples
        if len(opt.counterfactuals_iw):
            c_iws = opt.counterfactuals_iw.replace(" ", "").split(",")
            counterfactual_estimators_ids = [f"{e}-iw{k}" for e in counterfactual_estimators_ids for k in c_iws]

            print(">>> counterfactual_estimators_ids")
            print(counterfactual_estimators_ids)
            print("--------------------------------")

        def _split(e):
            """parse estimator name with key `-iw` else use opt.iw"""
            if "-iw" in e:
                arg = e.split("-")[-1]
                if arg == "iwae":  # catch special case
                    return e, opt.iw
                iw = eval(arg.replace("iw", ""))
                e = e.replace(f"-iw{iw}", "")
                return e, iw
            else:
                return e, opt.iw

        counterfactual_estimators, counterfactual_iw = zip(*[_split(e) for e in counterfactual_estimators_ids])
    else:
        counterfactual_estimators, counterfactual_iw, counterfactual_estimators_ids = [], [], []

    return counterfactual_estimators, counterfactual_iw, counterfactual_estimators_ids


def get_run_id(opt, counterfactual_estimators):
    """define a unique identifier given on the parsed config"""
    use_baseline = any(['-baseline' in e for e in [opt.estimator] + list(counterfactual_estimators)])
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
    if opt.oracle != "":
        run_id += f"-oracle={opt.oracle}-iw{opt.oracle_iw_samples}"
    if opt.model == 'bernoulli_toy':
        run_id += f"-tar{opt.toy_target}"
    if opt.skip:
        run_id += "-skip"
    if opt.warmup > 0:
        run_id += f"-{opt.warmup_mode}-warmup{opt.warmup}-{opt.gamma_min}-{opt.gamma}"
    else:
        if opt.gamma != 1:
            run_id += f"-gamma{opt.gamma}"

    exp_id = f"{opt.exp}-{opt.estimator}-g={opt.gamma}-K={opt.iw}"

    return run_id, exp_id, use_baseline


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


def init_model(opt, x, xmean=None):
    # hyperparameters
    hyperparams = {
        'xdim': x.shape,
        'N': opt.N,
        'K': opt.K,
        'hdim': opt.hdim,
        'kdim': opt.kdim,
        'nlayers': opt.nlayers,
        'depth': opt.depth,
        'learn_prior': opt.learn_prior,
        'prior': opt.prior,
        'normalization': opt.norm,
        'dropout': opt.dropout,
        'x_mean': xmean,
        'skip': opt.skip
    }

    # get the right constructor
    model_id = {'gmm-toy': 'gmm-toy',
                'gaussian-toy': 'gaussian-toy',
                'bernoulli-toy': 'bernoulli-toy',
                'air': 'air'}.get(opt.dataset, opt.model)

    _MODEL = {'vae': VAE,
              'conv-vae': ConvVAE,
              'bernoulli-toy': BernoulliToyModel,
              'gaussian-toy': GaussianToyVAE,
              'gmm-toy': GaussianMixture,
              'hierarchical': HierarchicalVae,
              'sbm': SigmoidBeliefNetwork,
              'air': AIR}[model_id]

    # init the model
    torch.manual_seed(opt.seed)
    return _MODEL(**hyperparams)


def init_neural_baseline(opt, x):
    """define a neural baseline"""
    return Baseline(x.shape, opt.b_nlayers, opt.hdim)


def init_main_estimator(opt, baseline):
    """initialize the training estimator and its configuration dict"""
    Estimator, config = get_config(opt.estimator)
    estimator = Estimator(baseline=baseline, mc=opt.mc, iw=opt.iw, N=opt.N, K=opt.K, hdim=opt.hdim,
                          freebits=opt.freebits, **config)

    # additionals arguments
    config.update({'beta': opt.beta})

    return estimator, config

def init_warnup_estimator(opt):
    """initialize the training estimator and its configuration dict"""
    if opt.warmup_estimator != "":
        Estimator, config = get_config(opt.warmup_estimator)
        estimator = Estimator(baseline=baseline, mc=opt.mc, iw=opt.iw, N=opt.N, K=opt.K, hdim=opt.hdim,
                              freebits=opt.freebits, **config)

        return estimator, config
    else:
        return None, None


def init_test_estimator(opt):
    """
    Initialize the 2 validation estimators.
    * a. estimator_valid: used to compute L_K_valid
    * b. estimator_valid_as_training: used to compute L_K_train

    The two estimators are used toe estimate KL(q||p) = log \hat{p(x)} - L_K_train considering log \hat{p(x)} = L_K_valid
    """
    Estimator = AirReinforce if 'air' == opt.dataset else VariationalInference
    config = {'tau': 0, 'zgrads': False}

    # the main test estimator with K = opt.iw_test
    estimator = Estimator(mc=1, iw=opt.iw_test, sequential_computation=opt.sequential_computation)
    estimator_ess = Estimator(mc=1, iw=opt.iw, sequential_computation=opt.sequential_computation)

    return estimator, estimator_ess, config


def init_oracle_estimator(opt):
    if len(opt.oracle):
        Estimator, config_oracle = get_config(opt.oracle)
        estimator_oracle = Estimator(mc=1, iw=opt.oracle_iw_samples)
    else:
        estimator_oracle = config_oracle = None

    return estimator_oracle, config_oracle


def init_counterfactual_estimators(opt, counterfactual_estimators, counterfactual_iw):
    """initialize the so called `counterfactual` estimators"""
    if len(counterfactual_estimators):
        c_Estimators, c_configs = zip(*[get_config(c) for c in counterfactual_estimators])
        c_estimators = [Est(baseline=baseline, mc=opt.mc, iw=c_iw, N=opt.N, K=opt.K, hdim=opt.hdim) for Est, c_iw in
                        zip(c_Estimators, counterfactual_iw)]
    else:
        c_configs = c_estimators = []

    return c_estimators, c_configs


def init_optimizers(opt, model, estimator):
    """Initialize estimators both for the model's parameters and the baselines/estimator's parameters"""
    optimizers = []
    _OPT = {'sgd': SGD, 'adam': Adam, 'adamax': Adamax, 'rmsprop': RMSprop}[opt.optimizer]
    optimizers += [_OPT(model.parameters(), lr=opt.lr)]
    if len(list(estimator.parameters())):
        optimizers += [torch.optim.Adam(estimator.parameters(), lr=opt.baseline_lr)]

    return optimizers


def get_number_of_epochs(opt):
    """define the number of training epochs based on the lenght of the dataset and the number of steps"""
    epochs = opt.epochs
    iter_per_epoch = len(loader_train.dataset) // opt.bs
    if epochs < 0:
        epochs = 1 + opt.nsteps // iter_per_epoch
    return epochs, iter_per_epoch


def get_grads_analysis_config(opt, model, x_eval, estimator_oracle, config_oracle):
    """
    Define the arguments for the gradients analysis and potentially compute the `oracle gradients`
    """

    grad_args = {'seed': opt.seed, 'batch_size': opt.bs * opt.mc * opt.iw,
                 'n_samples': opt.grad_samples,
                 'key_filter': opt.grad_key, 'use_individual_grads': opt.individual_grads,
                 'use_dsnr': True}

    # compute oracle gradients (the true gradients direction)
    if estimator_oracle is not None:
        print(
            f">> evaluating oracle `{opt.oracle}`, K = {opt.oracle_iw_samples} using {opt.oracle_mc_samples} MC samples")
        # define args
        oracle_args = copy(grad_args)
        seed = oracle_args.pop('seed', 13) + 1
        oracle_args.update({'seed': seed, 'n_samples': opt.oracle_mc_samples})

        _, oracle_grads = get_gradients_statistics(estimator_oracle, model, x_eval, **oracle_args,
                                                   **config_oracle)
        oracle_grads = oracle_grads['expected']
        oracle_grads = oracle_grads / oracle_grads.norm(p=2, dim=-1, keepdim=True)
        grad_args.update({'true_grads': oracle_grads})

    return grad_args


def perform_gradients_analysis(opt, global_step, writer_train, loader_train, model, estimator, config,
                               estimator_oracle=None, config_oracle=None,
                               counterfactual_writers=[], c_estimators=[], c_configs=[], counterfactual_estimators=[]):
    """
    Perform the gradients analysis for the main estimator and the counterfactual estimators if available
    """
    device = next(iter(model.parameters())).device

    # batch of data for the gradients variance evaluation (at maximum of size bs)
    batch = next(iter(loader_train))
    x, *_ = preprocess(batch, device)
    x = x[:opt.grad_bs]

    # get the configuration for the gradients analysis including computing the gradients for the oracle
    grad_args = get_grads_analysis_config(opt, model, x, estimator_oracle, config_oracle)

    # gradients analysis for the training estimator
    grad_data, _ = get_gradients_statistics(estimator, model, x, **grad_args, **config)
    with torch.no_grad():
        summary = Diagnostic(grad_data)
        summary.log(writer_train, global_step)
        train_logger.info(f"{exp_id} | grads | "
                          f"snr = {grad_data.get('grads', {}).get('snr', 0.):.3E}, "
                          f"variance = {grad_data.get('grads', {}).get('variance', 0.):.3E}, "
                          f"magnitude = {grad_data.get('grads', {}).get('magnitude', 0.):.3E}, "
                          f"dsnr = {grad_data.get('grads', {}).get('dsnr', -1):.3E}, "
                          f"direction = {grad_data.get('grads', {}).get('direction', -1):.3E}")

    # analsysis for the `counterfactual` estimators
    for (c_writer, c_conf, c_est, c) in zip(counterfactual_writers, c_configs, c_estimators,
                                            counterfactual_estimators):
        grad_data, _ = get_gradients_statistics(c_est, model, x, **grad_args, **c_conf)
        with torch.no_grad():
            summary = Diagnostic(grad_data)
            summary.log(c_writer, global_step)
            train_logger.info(
                f" | counterfactual | {c:{max(map(len, counterfactual_estimators))}s} K = {c_est.iw * c_est.mc} "
                f"| snr = {grad_data.get('grads', {}).get('snr', 0.):.3f}, "
                f"variance = {grad_data.get('grads', {}).get('variance', 0.):.3f}, "
                f"dsnr = {grad_data.get('grads', {}).get('dsnr', -1):.3E}, "
                f"direction = {grad_data.get('grads', {}).get('direction', -1):.3E}"
            )


@torch.no_grad()
def evaluation(model, estimator, config, loader, exp_id, device='cpu', ref_summary=None, seed=None, max_eval=None):
    """evaluation epoch to estimate the marginal log-likelihood: L_K \approx log \hat{p(x)}, K -> \infty"""

    if seed is not None:
        _seed = int(torch.randint(1, sys.maxsize, (1,)).item())
        torch.manual_seed(seed)

    k = 0
    model.eval()
    agg = Aggregator()
    for batch in tqdm(loader, desc=f"{exp_id}-K={estimator.iw}-M={estimator.mc}-eval"):
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

    if seed is not None:
        torch.manual_seed(_seed)

    return summary


def preprocess(batch, device):
    if isinstance(batch, Tensor):
        x = batch.to(device)
        return x, None
    else:
        x, y = batch  # assume tuple (x,y)
        x = x.to(device)
        y = y.to(device)
        return x, y


if __name__ == '__main__':
    _sep = os.get_terminal_size().columns * "-"

    opt = parse_args()

    # deterministic backend, silent mode and checking args
    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if opt.silent:
        tqdm = notqdm

    if opt.model == 'bernoulli_toy' or opt.dataset == 'bernoulli_toy':
        assert opt.model == 'bernoulli_toy'
        assert opt.bs == 1 and opt.valid_bs == 1 and opt.test_bs == 1

    # define couterfactual estimators if any
    counterfactual_estimators, counterfactual_iw, counterfactual_estimators_ids = define_counterfactuals(opt)

    # defining the run identifier
    run_id, exp_id, use_baseline = get_run_id(opt, counterfactual_estimators)

    # defining the run directory
    logdir = init_logging_directory(opt, run_id)

    # save run configuration to the log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fp:
        _opt = vars(opt)
        _opt['hostname'] = socket.gethostname()
        fp.write(json.dumps(_opt, default=lambda x: str(x), indent=4))

    # wrap the training loop inside a try/except so we can write potential errors to a file.
    try:
        # define logger
        base_logger, train_logger, valid_logger, test_logger = get_loggers(logdir)
        print(_sep)
        base_logger.info(f"Run id: {run_id}")
        base_logger.info(f"Logging directory: {logdir}")
        base_logger.info(f"Torch version: {torch.__version__}")
        print(_sep)
        # setting the random seed
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        # get datasets (ony use training sets if `opt.only_train_set`)
        dset_train, dset_valid, dset_test = get_datasets(opt)
        base_logger.info(f"Dataset size: train = {len(dset_train)}, valid = {len(dset_valid)}, test = {len(dset_test)}")

        # dataloaders
        loader_train = DataLoader(dset_train, batch_size=opt.bs, shuffle=True, num_workers=opt.workers, pin_memory=True)

        # test loaders
        loader_eval_train = DataLoader(dset_train, batch_size=opt.test_bs, shuffle=True, num_workers=1,
                                       pin_memory=False)
        loader_eval_test = DataLoader(dset_test, batch_size=opt.test_bs, shuffle=True, num_workers=1,
                                      pin_memory=False)

        # compute mean value of train set
        mean_over_dset = get_dataset_mean(loader_train)

        # get a sample to evaluate the input shape
        x = dset_train[0]
        if not isinstance(x, Tensor):
            x, *_ = x
        base_logger.info(
            f"Sample: x.shape = {x.shape}, x.min = {x.min():.1f}, x.max = {x.max():.1f}, "
            f"x.mean = {mean_over_dset.mean():1f}, x.dtype = {x.dtype}")

        model = init_model(opt, x, mean_over_dset)
        base_logger.info(f"Number of parameters = {sum(p.numel() for p in model.parameters()):.3E}")

        print(_sep)
        print("Parameters")
        print(_sep)
        for k, v in model.named_parameters():
            base_logger.info(f"{k} : N = {v.numel()}, mean = {v.mean().item():.3f}, std = {v.std().item():.3f}")

        # define a neural baseline that can be used for the different estimators
        baseline = init_neural_baseline(opt, x) if use_baseline else None

        # training estimator
        estimator, config = init_main_estimator(opt, baseline)

        # warmup estimator
        warmup_estimator, warmup_config = init_warnup_estimator(opt)

        # test estimator (it is important that all models are evaluated using the same evaluator)
        estimator_test, estimator_test_ess, config_test = init_test_estimator(opt)

        # oracle estimator to establish the true gradients direction
        estimator_oracle, config_oracle = init_oracle_estimator(opt)

        # counterfactual estimators:
        # they are use to measure the variance of the gradients given other estimator without using them for optimization
        c_estimators, c_configs = init_counterfactual_estimators(opt, counterfactual_estimators, counterfactual_iw)

        # get device and move models
        device = "cuda:0" if torch.cuda.device_count() else "cpu"
        model.to(device)
        estimator.to(device)

        # define the optimizer for the model's parameters and the training estimator's parameters if any (baseline)
        optimizers = init_optimizers(opt, model, estimator)

        # Deterministic warmup
        scheduler = LinearSchedule(opt.warmup, opt.gamma_min, opt.gamma,
                                   offset=opt.warmup_offset, mode=opt.warmup_mode) if opt.warmup > 0 else lambda x: opt.gamma

        # tensorboard writers used to log the summary
        writer_train = SummaryWriter(os.path.join(logdir, 'train'))
        writer_test = SummaryWriter(os.path.join(logdir, 'test'))
        counterfactual_writers = [SummaryWriter(os.path.join(logdir, c)) for c in counterfactual_estimators_ids]

        # define the run length based on either
        epochs, iter_per_epoch = get_number_of_epochs(opt)
        print(_sep)
        base_logger.info(f"Dataset = {opt.dataset}: running for {epochs} epochs,"
                         f" {iter_per_epoch * epochs} steps, {iter_per_epoch} steps / epoch, {epochs // opt.eval_freq} eval. steps")
        print(_sep)

        # sample model at initialization
        sample_model("prior-sample", model, logdir, global_step=0, writer=writer_test, seed=opt.seed)

        # run
        best_elbo = (-1e20, 0, 0)
        global_step = 0

        for epoch in range(1, epochs + 1):

            """training epoch"""
            [o.zero_grad() for o in optimizers]
            model.train()
            for batch in tqdm(loader_train, desc=f"{exp_id}-K={estimator.iw}-M={estimator.mc}-training"):

                # get the estimator & config
                if warmup_estimator is not None and global_step < opt.warmup:
                    _estimator, _config = warmup_estimator, warmup_config
                else:
                    _estimator, _config = estimator, config

                gamma = scheduler(global_step)
                x, y = preprocess(batch, device)
                training_step(x, model, _estimator, optimizers, y=y, return_diagnostics=False, gamma=gamma, **_config)
                global_step += 1

            if epoch % opt.eval_freq == 0:
                parameters_diagnostics = {'parameters' : {'beta': _config.get('beta', 1.), 'gamma': gamma}}

                """Total derivatives Analysis"""
                if opt.mc_analysis:
                    summary = Diagnostic(total_derivatives_analysis(estimator, model, x, opt.mc_analysis, **config))
                    summary.log(writer_train, global_step)

                """Active Units Analysis"""
                if opt.mc_au_analysis:
                    summary = Diagnostic(latent_activations(model, loader_eval_train, opt.mc_au_analysis, nsamples=opt.npoints_au_analysis, seed=opt.seed))
                    summary.log(writer_train, global_step)


                """Analyse Gradients"""
                if opt.grad_samples > 0:
                    print(_sep)
                    perform_gradients_analysis(opt, global_step, writer_train, loader_train, model, estimator, config,
                                               estimator_oracle=estimator_oracle, config_oracle=config_oracle,
                                               counterfactual_writers=counterfactual_writers, c_estimators=c_estimators,
                                               c_configs=c_configs, counterfactual_estimators=counterfactual_estimators)

                """Eval on the train set"""
                if not opt.only_train_set:
                    # log train summary to console and tensorboad
                    # seed evaluation such that the subset of `opt.max_eval` data points remains the same
                    summary_train = evaluation(model, estimator_test, config_test, loader_eval_train, exp_id,
                                               device=device, ref_summary=None, max_eval=opt.max_eval, seed=opt.seed)

                    # eval ess using the training estimator
                    summary_train_ = evaluation(model, estimator_test_ess, config_test, loader_eval_train, exp_id,
                                                device=device, ref_summary=None, max_eval=1000, seed=opt.seed)

                    summary_train['loss']['ess'] = summary_train_['loss']['ess']
                else:
                    # eval ess using the training estimator
                    summary_train = evaluation(model, estimator_test_ess, config_test, loader_eval_train, exp_id,
                                               device=device, ref_summary=None, max_eval=1000, seed=opt.seed)

                # log train summary to console and tensorboard
                summary_train.update(parameters_diagnostics)
                log_summary(summary_train, global_step, epoch, logger=train_logger, best=None,
                            writer=writer_train,
                            exp_id=exp_id)

                """eval on the test set"""
                summary_test = evaluation(model, estimator_test, config_test, loader_eval_test, exp_id, device=device,
                                          ref_summary=summary_train, max_eval=opt.max_eval)

                # update best elbo and save model
                best_elbo = save_model_and_update_best_elbo(model, summary_test, global_step,
                                                            epoch, best_elbo, logdir)

                # log marginal test summary to console and tensorboar
                summary_test.update(parameters_diagnostics)
                log_summary(summary_test, global_step, epoch, logger=test_logger, best=best_elbo,
                            writer=writer_test,
                            exp_id=exp_id)

                """sample model"""
                sample_model("prior-sample", model, logdir, global_step=global_step, writer=writer_test, seed=opt.seed)
                print(_sep)

            """reduce learning rate"""
            if opt.lr_reduce_steps > 0:
                lr_freq = (epochs // (opt.lr_reduce_steps + 1))
                if epoch % lr_freq == 0:
                    for o in optimizers:
                        for i, param_group in enumerate(o.param_groups):
                            lr = param_group['lr']
                            new_lr = lr / 2
                            param_group['lr'] = new_lr
                            base_logger.info(f"Reducing lr, group = {i} : {lr:.2E} -> {new_lr:.2E}")

        """final testing given the parameters from the best test score"""
        print(f"{_sep}\nFinal validation with best test "
              f"L_{opt.iw_test} = {best_elbo[0]:.3f} at step {best_elbo[1]}, epoch = {best_elbo[2]}\n{_sep}")

        writer_valid = SummaryWriter(os.path.join(logdir, 'valid'))
        loader_valid = DataLoader(dset_valid, batch_size=opt.test_bs, shuffle=True, num_workers=1)

        config_valid = {'tau': 0, 'zgrads': False}
        Estimator_valid = AirReinforce if 'air' == opt.dataset else VariationalInference
        estimator_valid = Estimator_valid(mc=1, iw=opt.iw_valid,
                                          sequential_computation=opt.test_sequential_computation)

        # load best model and run over the test set
        load_model(model, logdir)
        model.eval()
        agg = Aggregator()
        with torch.no_grad():
            for batch in tqdm(loader_valid, desc=f"{exp_id}-K={estimator_valid.iw}-M={estimator_valid.mc}"):
                x, y = preprocess(batch, device)
                diagnostics = test_step(x, model, estimator_valid, y=y, **config_test)
                agg.update(diagnostics)
            summary = agg.data.to('cpu')

        # log
        _, global_step, epoch = best_elbo
        log_summary(summary, global_step, epoch, logger=valid_logger, best=None, writer=writer_valid, exp_id=exp_id)

        # write outcome to a file (success, interrupted, error)
        print(f"{_sep}\nSucces.\n{_sep}")
        with open(os.path.join(logdir, "success.txt"), 'w') as f:
            f.write(f"Success.")

    except KeyboardInterrupt:
        print(f"{_sep}\nKeyboard Interrupt.\n{_sep}")
        with open(os.path.join(logdir, "success.txt"), 'w') as f:
            f.write(f"Failed. Interrupted (keyboard).")


    except Exception as ex:
        print(f"{_sep}\nFailed with exception {type(ex).__name__} = `{ex}` \n{_sep}")
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        with open(os.path.join(logdir, "success.txt"), 'w') as f:
            f.write(f"Failed. Exception : \n{ex}\n\n{ex.__traceback__}")
