import argparse
import json
import os
import socket
import traceback
from copy import copy
from shutil import rmtree

import numpy as np
import torch
from booster import Aggregator, Diagnostic
from torch.optim import Adam, Adamax, SGD, RMSprop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import get_datasets
from lib.config import get_config
from lib.estimators import VariationalInference, AirReinforce
from lib.gradients import get_gradients_statistics
from lib.logging import sample_model, get_loggers, log_summary, save_model, load_model
from lib.models import VAE, Baseline, ConvVAE, ToyVAE, GaussianMixture, AIR, HierarchicalVae
from lib.ops import training_step, test_step
from lib.utils import notqdm

if __name__ == '__main__':

    _sep = os.get_terminal_size().columns * "-"

    parser = argparse.ArgumentParser()

    # run directory, id and seed
    parser.add_argument('--dataset', default='shapes', help='dataset [shapes | binmnist | omniglot | fashion]')
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
    parser.add_argument('--test_bs', default=1, type=int, help='evaluation batch size')
    parser.add_argument('--grad_bs', default=10, type=int, help='grads evaluation batch size')
    parser.add_argument('--lr_reduce_steps', default=0, type=int, help='number of learning rate reduce steps')
    parser.add_argument('--only_train_set', action='store_true',
                        help='only use the training dataset: useful to study optimization behaviour.')

    # estimator
    parser.add_argument('--estimator', default='reinforce', help='[vi, reinforce, vimco, gs, st-gs]')
    parser.add_argument('--mc', default=1, type=int, help='number of Monte-Carlo samples')
    parser.add_argument('--iw', default=1, type=int, help='number of Importance-Weighted samples')
    parser.add_argument('--beta', default=1.0, type=float, help='Beta weight for the KL term (i.e. Beta-VAE)')

    # evaluation
    parser.add_argument('--eval_freq', default=10, type=int, help='frequency for the evaluation [test set + grads]')
    parser.add_argument('--iw_valid', default=100, type=int,
                        help='number of Importance-Weighted samples for validation')
    parser.add_argument('--iw_test', default=1000, type=int, help='number of Importance-Weighted samples for testing')

    # gradients analysis
    parser.add_argument('--grad_samples', default=100, type=int,
                        help='number of samples used to evaluate the variance.')
    parser.add_argument('--batch_grads', action='store_true',
                        help='Compute expected gradients over mini-batch instead of grads for single datapoints.')
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
    parser.add_argument('--model', default='vae', help='[vae, conv-vae, hierarchical]')
    parser.add_argument('--hdim', default=64, type=int, help='number of hidden units for each layer')
    parser.add_argument('--nlayers', default=3, type=int, help='number of hidden layers for the encoder and decoder')
    parser.add_argument('--depth', default=3, type=int,
                        help='number of stochastic layers when using hierarchical models')
    parser.add_argument('--b_nlayers', default=1, type=int, help='number of hidden layers for the baseline')
    parser.add_argument('--norm', default='layernorm', type=str,
                        help='normalization layer [none | layernorm | batchnorm]')
    parser.add_argument('--dropout', default=0, type=float, help='dropout value')

    opt = parser.parse_args()

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if opt.silent:
        tqdm = notqdm

    # define conunterfactuals
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

    # defining the run identifier
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
    run_id += f"-arch{opt.hdim}x{opt.nlayers}"
    if opt.norm is not 'none':
        run_id += f"-{opt.norm}"
    if opt.dropout > 0:
        run_id += f"-drp{opt.dropout}"
    if opt.oracle != "":
        run_id += f"-oracle={opt.oracle}-iw{opt.oracle_iw_samples}"

    # _exp_id = f"{opt.exp}-{opt.dataset}-{opt.estimator}"
    _exp_id = f"{opt.exp}-{opt.estimator}"

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
        _opt = vars(opt)
        _opt['hostname'] = socket.gethostname()
        fp.write(json.dumps(_opt, default=lambda x: str(x), indent=4))

    # wrap the training loop inside a try/except so we can write potential errors to a file.
    try:
        # define logger
        base_logger, train_logger, valid_logger, test_logger = get_loggers(logdir)
        base_logger.info(f"Run id: {run_id}")
        base_logger.info(f"Logging directory: {logdir}")
        base_logger.info(f"Torch version: {torch.__version__}")

        # setting the random seed
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        # get datasets (ony use training sets if `opt.only_train_set`)
        dset_train, dset_valid, dset_test = get_datasets(opt)
        base_logger.info(f"Dataset size: train = {len(dset_train)}, valid = {len(dset_valid)}")

        # dataloaders
        loader_train = DataLoader(dset_train, batch_size=opt.bs, shuffle=True, num_workers=opt.workers, pin_memory=True)
        loader_valid = DataLoader(dset_valid, batch_size=opt.valid_bs, shuffle=True, num_workers=opt.workers,
                                  pin_memory=True)

        # compute mean value of train set
        _xmean = None
        _n = 0.
        for x in loader_train:
            k = x.size(0)
            m = x.sum(0)
            _n += k
            if _xmean is None:
                _xmean = m / k
            else:
                _xmean += (m - k * _xmean) / _n
        _xmean = _xmean.unsqueeze(0)

        # get a sample to evaluate the input shape
        x = dset_train[0]
        base_logger.info(
            f"Sample: x.shape = {x.shape}, x.min = {x.min():.1f}, x.max = {x.max():.1f}, x.mean = {_xmean.mean():1f}, x.dtype = {x.dtype}")

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
            'x_mean': _xmean
        }
        # get the right constructor
        model_id = {'gmm': 'gmm', 'gaussian-toy': 'toy-vae', 'air': 'air'}.get(opt.dataset, opt.model)
        _MODEL = {'vae': VAE,
                  'conv-vae': ConvVAE,
                  'toy-vae': ToyVAE,
                  'gmm': GaussianMixture,
                  'air': AIR,
                  'hierarchical': HierarchicalVae}[model_id]

        # init model
        torch.manual_seed(opt.seed)
        model = _MODEL(**hyperparams)

        # define baseline
        baseline = Baseline(x.shape, opt.b_nlayers, opt.hdim) if use_baseline else None

        # estimator
        Estimator, config = get_config(opt.estimator)
        estimator = Estimator(baseline=baseline, mc=opt.mc, iw=opt.iw, N=opt.N, K=opt.K, hdim=opt.hdim,
                              freebits=opt.freebits)

        # add beta parameter to the config
        config.update({'beta': opt.beta})

        # valid estimator (it is important that all models are evaluated using the same evaluator)
        Estimator_valid = AirReinforce if 'air' == opt.dataset else VariationalInference
        config_valid = {'tau': 0, 'zgrads': False}
        estimator_valid = Estimator_valid(mc=1, iw=opt.iw_valid, sequential_computation=opt.sequential_computation)

        # oracle estimator to find the true gradients direction
        if len(opt.oracle):
            Estimator, config_oracle = get_config(opt.oracle)
            estimator_oracle = Estimator(mc=1, iw=opt.oracle_iw_samples)
        else:
            estimator_oracle = None

        # counterfactual estimators:
        # they are use to measure the variance of the gradients given other estimator without using them for optimization
        if len(counterfactual_estimators):
            c_Estimators, c_configs = zip(*[get_config(c) for c in counterfactual_estimators])
            c_estimators = [Est(baseline=baseline, mc=opt.mc, iw=c_iw, N=opt.N, K=opt.K, hdim=opt.hdim) for Est, c_iw in
                            zip(c_Estimators, counterfactual_iw)]
        else:
            c_configs = c_estimators = []

        # get device and move models
        device = "cuda:0" if torch.cuda.device_count() else "cpu"
        model.to(device)
        estimator.to(device)
        estimator_valid.to(device)

        # optimizers
        optimizers = []
        _OPT = {'sgd': SGD, 'adam': Adam, 'adamax': Adamax, 'rmsprop': RMSprop}[opt.optimizer]
        optimizers += [_OPT(model.parameters(), lr=opt.lr)]
        if len(list(estimator.parameters())):
            optimizers += [torch.optim.Adam(estimator.parameters(), lr=opt.baseline_lr)]

        # print(f"{_sep}\nModel paramters:")
        # for k, v in model.named_parameters():
        #     print(f"   {k} : {v.numel()}")
        # print(f"{_sep}\nEstimator paramters:")
        # for k, v in estimator.named_parameters():
        #     print(f"   {k} : {v.numel()}")
        # print(_sep)

        # data aggregator
        agg_train = Aggregator()
        agg_valid = Aggregator()

        # tensorboard writers
        writer_train = SummaryWriter(os.path.join(logdir, 'train'))
        writer_valid = SummaryWriter(os.path.join(logdir, 'valid'))
        counterfactual_writers = [SummaryWriter(os.path.join(logdir, c)) for c in counterfactual_estimators_ids]

        # run lenght
        epochs = opt.epochs
        iter_per_epoch = len(loader_train.dataset) // opt.bs
        if epochs < 0:
            print("# iter_per_epoch:", iter_per_epoch)
            epochs = 1 + opt.nsteps // iter_per_epoch
        base_logger.info(f"# {opt.dataset}: running for {epochs} epochs, {iter_per_epoch * epochs} steps")

        # sample model
        sample_model("prior-sample", model, logdir, global_step=0, writer=writer_valid, seed=opt.seed)

        # run
        best_elbo = (-1e20, 0, 0)
        global_step = 0
        for epoch in range(1, epochs + 1):

            # train epoch
            [o.zero_grad() for o in optimizers]
            model.train()
            agg_train.initialize()
            for x in tqdm(loader_train, desc=_exp_id):
                x = x.to(device)
                diagnostics = training_step(x, model, estimator, optimizers, **config)
                agg_train.update(diagnostics)

                global_step += 1
            summary_train = agg_train.data.to('cpu')

            if epoch % opt.eval_freq == 0:

                if opt.grad_samples > 0:

                    # batch of data for the gradients variance evaluation (at maximum of size bs)
                    x_eval = next(iter(loader_train)).to(device)[:opt.grad_bs]

                    # estimate the variance of the gradients
                    grad_args = {'seed': opt.seed, 'batch_size': opt.bs * opt.mc * opt.iw,
                                 'n_samples': opt.grad_samples,
                                 'key_filter': 'tensor:qlogits', 'use_batch_grads': opt.batch_grads, 'use_dsnr': True}

                    # compute oracle gradients (the true gradients direction)
                    if estimator_oracle is not None:
                        print(
                            f">> evaluating oracle `{opt.oracle}`, K = {opt.oracle_iw_samples} using {opt.oracle_mc_samples} MC samples (batch_grads = {opt.batch_grads})")
                        # define args
                        oracle_args = copy(grad_args)
                        seed = oracle_args.pop('seed', 13) + 1
                        oracle_args.update({'seed': seed, 'n_samples': opt.oracle_mc_samples})

                        _, oracle_grads = get_gradients_statistics(estimator_oracle, model, x_eval, **oracle_args,
                                                                   **config_oracle)
                        oracle_grads = oracle_grads['expected']
                        oracle_grads = oracle_grads / oracle_grads.norm(p=2, dim=-1, keepdim=True)
                        grad_args.update({'true_grads': oracle_grads})

                    # for the main estimator
                    grad_data, _ = get_gradients_statistics(estimator, model, x_eval, **grad_args, **config)
                    with torch.no_grad():
                        summary_train.update(grad_data)
                        train_logger.info(f"{_exp_id} | grads | "
                                          f"snr = {grad_data.get('grads', {}).get('snr', 0.):.3E}, "
                                          f"variance = {grad_data.get('grads', {}).get('variance', 0.):.3E}, "
                                          f"magnitude = {grad_data.get('grads', {}).get('magnitude', 0.):.3E}, "
                                          f"dsnr = {grad_data.get('grads', {}).get('dsnr', -1):.3E}, "
                                          f"direction = {grad_data.get('grads', {}).get('direction', -1):.3E}")

                    # analyse the control variate terms on the counter-factual estimators
                    # analyse_control_variate(x_eval, model, c_configs, c_estimators, counterfactual_estimators, writer_train, seed=opt.seed, global_step=global_step)

                    # counter-factual estimation of the other estimators
                    for (c_writer, c_conf, c_est, c) in zip(counterfactual_writers, c_configs, c_estimators,
                                                            counterfactual_estimators):
                        grad_data, _ = get_gradients_statistics(c_est, model, x_eval, **grad_args, **c_conf)
                        with torch.no_grad():
                            summary = Diagnostic(grad_data)
                            summary.log(c_writer, global_step)
                            train_logger.info(f" | counterfactual | {c:{max(map(len, counterfactual_estimators))}s} "
                                              f"| snr = {grad_data.get('grads', {}).get('snr', 0.):.3f}, "
                                              f"variance = {grad_data.get('grads', {}).get('variance', 0.):.3f}, "
                                              f"dsnr = {grad_data.get('grads', {}).get('dsnr', -1):.3E}, "
                                              f"direction = {grad_data.get('grads', {}).get('direction', -1):.3E}"
                                              )

                # validation epoch
                with torch.no_grad():
                    model.eval()
                    agg_valid.initialize()
                    for x in tqdm(loader_valid, desc=_exp_id):
                        x = x.to(device)
                        diagnostics = test_step(x, model, estimator_valid, **config)
                        agg_valid.update(diagnostics)
                    summary_valid = agg_valid.data.to('cpu')

                # update best elbo and save model
                best_elbo = save_model(model, summary_valid, global_step, epoch, best_elbo, logdir)

                # log valid summary to console and tensorboar
                log_summary(summary_valid, global_step, epoch, logger=valid_logger, best=best_elbo, writer=writer_valid,
                            exp_id=_exp_id)

                # sample model
                sample_model("prior-sample", model, logdir, global_step=global_step, writer=writer_valid, seed=opt.seed)

            # log train summary to console and tensorboar
            log_summary(summary_train, global_step, epoch, logger=train_logger, writer=writer_train, exp_id=_exp_id)

            # reduce learning rate
            if opt.lr_reduce_steps > 0:
                lr_freq = (epochs // (opt.lr_reduce_steps + 1))
                if epoch % lr_freq == 0:
                    for o in optimizers:
                        for i, param_group in enumerate(o.param_groups):
                            lr = param_group['lr']
                            new_lr = lr / 2
                            param_group['lr'] = new_lr
                            base_logger.info(f"Reducing lr, group = {i} : {lr:.2E} -> {new_lr:.2E}")

        print("TESTING..", best_elbo)
        # testing step
        writer_test = SummaryWriter(os.path.join(logdir, 'test'))
        loader_test = DataLoader(dset_test, batch_size=opt.test_bs, shuffle=True, num_workers=1)
        agg_test = Aggregator()

        config_test = {'tau': 0, 'zgrads': False}
        estimator_test = Estimator_valid(mc=1, iw=opt.iw_test,
                                         sequential_computation=opt.test_sequential_computation)

        # load best model and run over the test set
        load_model(model, logdir)
        model.eval()
        agg_test.initialize()
        with torch.no_grad():
            for x in tqdm(loader_test, desc=_exp_id + "-test"):
                x = x.to(device)
                diagnostics = test_step(x, model, estimator_test, **config_test)
                agg_test.update(diagnostics)
            summary_test = agg_test.data.to('cpu')

        # log
        _, global_step, epoch = best_elbo
        log_summary(summary_test, global_step, epoch, logger=test_logger, best=None, writer=writer_test, exp_id=_exp_id)

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
