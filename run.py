import argparse
import json
import os
import pickle
import traceback

import numpy as np
import torch
from booster import Aggregator, Diagnostic
from booster.utils import logging_sep, available_device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ovis import get_datasets
from ovis.analysis.active_units import latent_activations
from ovis.estimators import VariationalInference
from ovis.training.evaluation import analyse_gradients_and_log, evaluation
from ovis.training.initialization import init_model, init_neural_baseline, init_main_estimator, init_test_estimator, \
    init_optimizers, init_logging_directory
from ovis.training.logging import sample_prior_and_save_img, get_loggers, log_summary, save_model_and_update_best_elbo, load_model
from ovis.training.ops import training_step, test_step
from ovis.training.schedule import Schedule
from ovis.training.session import Session
from ovis.training.utils import get_run_id, get_number_of_epochs, preprocess
from ovis.utils.utils import notqdm, ManualSeed


def parse_args():
    """Parse the command line arguments"""

    parser = argparse.ArgumentParser()

    # run directory, id and seed
    parser.add_argument('--dataset', default='shapes',
                        help='dataset [shapes | binmnist | omniglot | fashion | gmm-toy | gaussian-toy]')
    parser.add_argument('--mini', action='store_true', help='use a sub-sampled version of the dataset')
    parser.add_argument('--root', default='runs/', help='directory to store training logs')
    parser.add_argument('--data_root', default='data/', help='directory to store the data')
    parser.add_argument('--exp', default='sandbox', help='experiment directory')
    parser.add_argument('--id', default='', type=str, help='run id suffix')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--workers', default=1, type=int, help='number of dataloader workers')
    parser.add_argument('--rm', action='store_true', help='delete the previous run')
    parser.add_argument('--silent', action='store_true', help='silence tqdm')
    parser.add_argument('--deterministic', action='store_true', help='use deterministic backend')
    parser.add_argument('--sequential_computation', action='store_true',
                        help='compute each iw sample sequentially during validation')
    parser.add_argument('--test_sequential_computation', action='store_true',
                        help='compute each iw sample sequentially during validation')

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
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha weight for the unormalized weights: log w_k^alpha')
    parser.add_argument('--warmup', default=0, type=int, help='period of the posterior warmup (alpha : 0 -> 1)')
    parser.add_argument('--warmup_offset', default=0, type=int, help='number of steps before increasing beta')
    parser.add_argument('--warmup_mode', default='log', type=str, help='interpolation mode [linear | log]')
    parser.add_argument('--alpha_min', default=0.9, type=float, help='minimum alpha value')

    # evaluation
    parser.add_argument('--eval_freq', default=10, type=int, help='frequency for the evaluation [test set + grads]')
    parser.add_argument('--max_eval', default=None, type=int, help='maximum number of data points for evaluation')
    parser.add_argument('--iw_valid', default=100, type=int,
                        help='number of Importance-Weighted samples for validation')
    parser.add_argument('--iw_test', default=1000, type=int, help='number of Importance-Weighted samples for testing')

    # active units analysis
    parser.add_argument('--mc_au_analysis', default=0, type=int,
                        help='number of Monte-Carlo samples used to estimate the number of active units')
    parser.add_argument('--npoints_au_analysis', default=1000, type=int,
                        help='number of data points x')

    # gradients analysis
    parser.add_argument('--grad_samples', default=100, type=int,
                        help='number of samples used to evaluate the variance.')
    parser.add_argument('--grad_key', default='inference_network', type=str,
                        help='identifiant of the parameters/tensor for the gradients analysis')

    # model architecture
    parser.add_argument('--model', default='vae', help='[vae, conv-vae, gmm, gaussian-toy, sbm, gaussian-vae]')
    parser.add_argument('--hdim', default=64, type=int, help='number of hidden units for each layer')
    parser.add_argument('--nlayers', default=3, type=int, help='number of hidden layers for the encoder and decoder')
    parser.add_argument('--depth', default=3, type=int,
                        help='number of stochastic layers when using hierarchical models')
    parser.add_argument('--b_nlayers', default=1, type=int, help='number of hidden layers for the neural baseline')
    parser.add_argument('--norm', default='none', type=str, help='normalization layer [none | layernorm | batchnorm]')
    parser.add_argument('--dropout', default=0, type=float, help='dropout value')
    parser.add_argument('--prior', default='normal', help='prior for the VAE model [normal, categorical, bernoulli]')
    parser.add_argument('--N', default=32, type=int, help='number of latent variables')
    parser.add_argument('--K', default=8, type=int, help='number of categories when using a categorical prior')
    parser.add_argument('--kdim', default=0, type=int, help='dimension of the keys model for categorical prior')
    parser.add_argument('--learn_prior', action='store_true', help='learn the prior')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    # deterministic backend, silent mode and checking args
    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if opt.silent:
        tqdm = notqdm

    # defining the run identifiers
    run_id, exp_id = get_run_id(opt)

    # defining the run directory
    logdir = init_logging_directory(opt, run_id)

    # save run configuration to the log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fp:
        fp.write(json.dumps(vars(opt), default=lambda x: str(x), indent=4))

    # wrap the training loop inside a try/except so we can write potential errors to a file.
    try:
        # define logger
        base_logger, train_logger, valid_logger, test_logger = get_loggers(logdir)
        print(logging_sep("="))
        base_logger.info(f"Run id: {run_id}")
        base_logger.info(f"Logging directory: {logdir}")
        base_logger.info(f"Torch version: {torch.__version__}")
        print(logging_sep("="))

        # setting the random seed
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)

        # get datasets (ony use training sets if `opt.only_train_set == True`)
        dset_train, dset_valid, dset_test = get_datasets(opt)
        base_logger.info(f"Dataset size: train = {len(dset_train)}, valid = {len(dset_valid)}, test = {len(dset_test)}")

        # training dataloader
        loader_train = DataLoader(dset_train, batch_size=opt.bs, shuffle=True, num_workers=opt.workers, pin_memory=True)

        # evaluation loaders
        loader_eval_train = DataLoader(dset_train, batch_size=opt.test_bs, shuffle=True, num_workers=1,
                                       pin_memory=False)
        loader_eval_test = DataLoader(dset_test, batch_size=opt.test_bs, shuffle=True, num_workers=1, pin_memory=False)

        # get a sample to evaluate the input shape
        x = dset_train[0]
        if not isinstance(x, Tensor):
            x, *_ = x
        base_logger.info(
            f"Sample: x.shape = {x.shape}, x.min = {x.min():.1f}, x.max = {x.max():.1f}, x.dtype = {x.dtype}")

        model, hyperparameters = init_model(opt, x, loader_train)
        # save hyper parameters for easy loading
        pickle.dump(hyperparameters, open(os.path.join(logdir, "hyperparameters.pkl"), "wb"))

        print(logging_sep("="))
        print(f"Parameters ({sum(p.numel() for p in model.parameters()):.3E})")
        print(logging_sep())
        for k, v in model.named_parameters():
            base_logger.info(f"{k} : N = {v.numel()}, mean = {v.mean().item():.3f}, std = {v.std().item():.3f}")
        print(logging_sep("="))

        # define a neural baseline that can be used for the different estimators
        baseline = init_neural_baseline(opt, x) if '-baseline' in opt.estimator else None

        # training estimator
        estimator, config = init_main_estimator(opt, baseline)

        # test estimator (it is important that all models are evaluated using the same evaluator)
        estimator_test, estimator_test_ess, config_test = init_test_estimator(opt)

        # get device and move models to device
        device = available_device()
        model.to(device)
        estimator.to(device)

        # define the optimizer for the model's parameters and the training estimator's parameters if any (baseline)
        optimizers = init_optimizers(opt, model, estimator)

        # Rényi warmup
        if opt.warmup > 0:
            scheduler = Schedule(opt.warmup, opt.alpha_min, opt.alpha, offset=opt.warmup_offset, mode=opt.warmup_mode)
        else:
            scheduler = lambda x: opt.alpha

        # tensorboard writers used to log the summary
        writer_train = SummaryWriter(os.path.join(logdir, 'train'))
        writer_test = SummaryWriter(os.path.join(logdir, 'test'))

        # define the run length based on either the number of epochs of number of steps
        epochs, iter_per_epoch = get_number_of_epochs(opt, loader_train)
        base_logger.info(f"Dataset = {opt.dataset}: running for {epochs} epochs, {iter_per_epoch * epochs} steps, "
                         f"{iter_per_epoch} steps / epoch, {epochs // opt.eval_freq} eval. steps\n{logging_sep()}")

        # sample model at initialization
        sample_prior_and_save_img("prior-sample", model, logdir, global_step=0, writer=writer_test, seed=opt.seed)

        # define the session and restore checkpoint if available
        session = Session(run_id, logdir, model, estimator, optimizers)
        session.restore_if_available()
        if session.epoch > 0:
            print(f"Restoring Session from epoch = {session.epoch} (best test "
                  f"L_{opt.iw_test} = {session.best_elbo[0]:.3f} at step {session.best_elbo[1]}, "
                  f"epoch = {session.best_elbo[2]})\n{logging_sep()}")

        # run
        while session.epoch < epochs:
            session.epoch += 1

            """training epoch"""
            [o.zero_grad() for o in optimizers]
            model.train()
            for batch in tqdm(loader_train, desc=f"{exp_id}-training"):
                alpha = scheduler(session.global_step)
                x, y = preprocess(batch, device)
                training_step(x, model, estimator, optimizers, y=y, return_diagnostics=False, alpha=alpha, **config)
                session.global_step += 1

            """reduce learning rate"""
            if opt.lr_reduce_steps > 0:
                lr_freq = (epochs // (opt.lr_reduce_steps + 1))
                if session.epoch % lr_freq == 0:
                    for o in optimizers:
                        for i, param_group in enumerate(o.param_groups):
                            lr = param_group['lr']
                            new_lr = lr / 2
                            param_group['lr'] = new_lr
                            base_logger.info(f"Reducing lr, group = {i} : {lr:.2E} -> {new_lr:.2E}")

            if session.epoch % opt.eval_freq == 0:
                parameters_diagnostics = {'parameters': {'beta': config.get('beta', 1.), 'alpha': alpha}}

                """Active Units Analysis"""
                if opt.mc_au_analysis:
                    summary = Diagnostic(latent_activations(model, loader_eval_train, opt.mc_au_analysis,
                                                            max_samples=opt.npoints_au_analysis))
                    summary.log(writer_train, session.global_step)

                """Analyse Gradients"""
                if opt.grad_samples > 0:
                    print(logging_sep())
                    analyse_gradients_and_log(opt, session.global_step, writer_train, train_logger, loader_train,
                                              model, estimator, config, exp_id, tqdm=tqdm)

                """Estimate the ESS"""
                summary_train_ess = evaluation(model, estimator_test_ess, config_test, loader_eval_train, exp_id,
                                               device=device, ref_summary=None, max_eval=1000, tqdm=tqdm)

                """Eval on the train set and logging"""
                if opt.only_train_set:
                    # if the test evaluation is performed on the test set, keep the summary from the evaluation
                    summary_train = summary_train_ess
                else:
                    # seed evaluation such that the subset of `opt.max_eval` data points remains the same
                    with ManualSeed(seed=opt.seed):
                        summary_train = evaluation(model, estimator_test, config_test, loader_eval_train, exp_id,
                                                   device=device, ref_summary=None, max_eval=opt.max_eval, tqdm=tqdm)

                    summary_train['loss']['ess'] = summary_train_ess['loss']['ess']

                # log train summary to console and tensorboard
                summary_train.update(parameters_diagnostics)
                log_summary(summary_train, session.global_step, session.epoch, logger=train_logger, best=None,
                            writer=writer_train,
                            exp_id=exp_id)

                """evaluation on the test set and logging"""
                summary_test = evaluation(model, estimator_test, config_test, loader_eval_test, exp_id, device=device,
                                          ref_summary=summary_train, max_eval=opt.max_eval, tqdm=tqdm)

                # update best elbo and save model
                session.best_elbo = save_model_and_update_best_elbo(model, summary_test, session.global_step,
                                                                    session.epoch, session.best_elbo, logdir)

                # log marginal test summary to console and tensorboar
                summary_test.update(parameters_diagnostics)
                log_summary(summary_test, session.global_step, session.epoch, logger=test_logger,
                            best=session.best_elbo,
                            writer=writer_test,
                            exp_id=exp_id)

                """sample model"""
                with ManualSeed(seed=opt.seed):
                    sample_prior_and_save_img("prior-sample", model, logdir, global_step=session.global_step, writer=writer_test)
                print(logging_sep())

                """Checkpointing"""
                session.save()

        """final testing given the parameters from the best test score"""
        print(f"{logging_sep()}\nFinal validation with best test "
              f"L_{opt.iw_test} = {session.best_elbo[0]:.3f} at step {session.best_elbo[1]}, epoch = {session.best_elbo[2]}\n{logging_sep()}")

        writer_valid = SummaryWriter(os.path.join(logdir, 'valid'))
        loader_valid = DataLoader(dset_valid, batch_size=opt.test_bs, shuffle=True, num_workers=1)

        config_valid = {'tau': 0, 'zgrads': False}
        Estimator_valid = VariationalInference
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
        _, global_step, epoch = session.best_elbo
        log_summary(summary, global_step, epoch, logger=valid_logger, best=None, writer=writer_valid, exp_id=exp_id)

        # write outcome to a file (success, interrupted, error)
        print(f"{logging_sep('=')}\nSucces.\n{logging_sep('=')}")
        with open(os.path.join(logdir, "success.txt"), 'w') as f:
            f.write(f"Success.")

    except KeyboardInterrupt:
        print(f"{logging_sep()}\nKeyboard Interrupt.\n{logging_sep()}")
        with open(os.path.join(logdir, "success.txt"), 'w') as f:
            f.write(f"Failed. Interrupted (keyboard).")


    except Exception as ex:
        print(f"{logging_sep()}\nFailed with exception {type(ex).__name__} = `{ex}` \n{logging_sep()}")
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        with open(os.path.join(logdir, "success.txt"), 'w') as f:
            f.write(f"Failed. Exception : \n{ex}\n\n{ex.__traceback__}")
