import json
import json
import os
import pickle

import numpy as np
import torch
from booster import Aggregator, Diagnostic
from booster.utils import logging_sep, available_device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ovis import get_datasets
from ovis.analysis.active_units import latent_activations
from ovis.estimators import VariationalInference
from ovis.training.arguments import *
from ovis.training.evaluation import analyse_gradients_and_log, evaluation
from ovis.training.initialization import init_model, init_neural_baseline, init_main_estimator, init_test_estimator, \
    init_optimizers, init_logging_directory
from ovis.training.logging import sample_prior_and_save_img, get_loggers, log_summary, save_model_and_update_best_elbo, \
    load_model
from ovis.training.ops import training_step, test_step
from ovis.training.schedule import Schedule
from ovis.training.session import Session
from ovis.training.utils import get_run_id, get_number_of_epochs, preprocess, reduce_lr
from ovis.utils.success import Success
from ovis.utils.utils import notqdm, ManualSeed, print_info


def run():
    """
    Learn the parameters of the specified model and evaluate. Evaluation is performed every `opt['eval_freq']` epochs
    on the test set using the `estimator_test`. At each evaluation step is also performed:
        * gradient analysis
        * measure the number of active units
        * sampling from the prior
        * evaluation on a subset of the training dataset
        * checkpointing best on `log p(x) = L_K (test)`

    A final evaluation step is performed on the `validation` set using the parameters of the model that scored the
    highest `L_K (test)`.

    Each session is identified by a unique deterministic `run_id`. Starting a novel session that matches an existing
    `run_id` will result in loading the last checkpoint from the existing session.
    """
    parser = argparse.ArgumentParser()
    add_base_args(parser, exp='sandbox')
    add_run_args(parser)
    add_model_architecture_args(parser)
    add_active_units_args(parser)
    add_gradient_analysis_args(parser)
    opt = vars(parser.parse_args())

    # deterministic backend and silent mode
    if opt['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if opt['silent']:
        tqdm = notqdm
    else:
        from tqdm import tqdm

    # defining the run identifiers
    run_id, exp_id, hash = get_run_id(opt)

    # defining the run directory
    logdir = init_logging_directory(opt, run_id)

    # save run configuration to the log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fp:
        opt['hash'] = hash
        fp.write(json.dumps(opt, default=lambda x: str(x), indent=4))

    # wrap the training loop inside with `Success` to write the outcome of the run to a file
    with Success(logdir=logdir):
        # get the device (cuda/cpu)
        device = available_device()

        # define logger
        base_logger, train_logger, valid_logger, test_logger = get_loggers(logdir)
        print_info(logdir=logdir, device=device, run_id=run_id, logger=base_logger)

        # setting the random seed
        torch.manual_seed(opt['seed'])
        np.random.seed(opt['seed'])

        # get datasets (ony use training sets if `opt['only_train_set'] == True`)
        dset_train, dset_valid, dset_test = get_datasets(opt)
        base_logger.info(f"Dataset size: train = {len(dset_train)}, valid = {len(dset_valid)}, test = {len(dset_test)}")

        # training dataloader
        loader_train = DataLoader(dset_train, batch_size=opt['bs'], shuffle=True, num_workers=opt['workers'],
                                  pin_memory=True)

        # evaluation loaders
        loader_eval_train = DataLoader(dset_train, batch_size=opt['test_bs'], shuffle=True, num_workers=1,
                                       pin_memory=False)
        loader_eval_test = DataLoader(dset_test, batch_size=opt['test_bs'], shuffle=True, num_workers=1,
                                      pin_memory=False)

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
        base_logger.info(f"Parameters (N = {sum(p.numel() for p in model.parameters()):.3E})")
        print(logging_sep())
        for k, v in model.named_parameters():
            base_logger.info(f"{k} : N = {v.numel()}, mean = {v.mean().item():.3f}, std = {v.std().item():.3f}")
        print(logging_sep("="))

        # define a neural baseline that can be used for the different estimators
        baseline = init_neural_baseline(opt, x) if '-baseline' in opt['estimator'] else None

        # training estimator
        estimator = init_main_estimator(opt, baseline=baseline)

        # test estimator (it is important that all models are evaluated using the same evaluator)
        estimator_test, estimator_test_ess = init_test_estimator(opt)

        # move models to device
        model.to(device)
        estimator.to(device)

        # define the optimizer for the model's parameters and the training estimator's parameters if any (baseline)
        optimizers = init_optimizers(opt, model, estimator)

        # parameters
        parameters = {
            'alpha': opt['alpha'],
            'beta': opt['beta'],
            'tau': opt['tau'],
            'freebits': opt['freebits']
        }
        # filters NaNs values from the parameters (otherwise Tensorboard logging throws an error)
        parameters = {k: v for k, v in parameters.items() if v is not None}

        # Rényi warmup
        if opt['warmup'] > 0:
            scheduler = Schedule(opt['warmup'], opt['alpha_init'], opt['alpha'], offset=opt['warmup_offset'],
                                 mode=opt['warmup_mode'])
        else:
            scheduler = lambda x: opt['alpha']

        # tensorboard writers used to log the summary
        writer_train = SummaryWriter(os.path.join(logdir, 'train'))
        writer_test = SummaryWriter(os.path.join(logdir, 'test'))

        # define the run length based on either the number of epochs of number of steps
        epochs, iter_per_epoch = get_number_of_epochs(opt, loader_train)
        base_logger.info(f"Dataset = {opt['dataset']}: running for {epochs} epochs, {iter_per_epoch * epochs} steps, "
                         f"{iter_per_epoch} steps / epoch, {epochs // opt['eval_freq']} eval. steps\n{logging_sep()}")

        # sample model at initialization
        sample_prior_and_save_img("prior-sample", model, logdir, global_step=0, writer=writer_test, seed=opt['seed'])

        # define the session and restore checkpoint if available
        session = Session(run_id, logdir, model, estimator, optimizers)
        session.restore_if_available()
        if session.epoch > 0:
            print(f"Restoring Session from epoch = {session.epoch} (best test "
                  f"L_{opt['iw_test']} = {session.best_elbo[0]:.3f} at step {session.best_elbo[1]}, "
                  f"epoch = {session.best_elbo[2]})\n{logging_sep()}")

        # run
        while session.epoch < epochs:
            session.epoch += 1

            """training epoch"""
            [o.zero_grad() for o in optimizers]
            model.train()
            for batch in tqdm(loader_train, desc=f"[training] {exp_id}"):
                parameters['alpha'] = scheduler(session.global_step)
                x, y = preprocess(batch, device)
                training_step(x, model, estimator, optimizers, y=y, return_diagnostics=False, **parameters)
                session.global_step += 1

            """reduce learning rate"""
            if opt['lr_reduce_steps'] > 0:
                reduce_lr(optimizers, session.epoch, epochs, opt['lr_reduce_steps'], base_logger)

            if session.epoch % opt['eval_freq'] == 0:
                parameters_diagnostics = {'parameters': parameters}

                """Active Units Analysis"""
                if opt['mc_au_analysis']:
                    summary = Diagnostic(latent_activations(model, loader_eval_train, opt['mc_au_analysis'],
                                                            max_samples=opt['npoints_au_analysis']))
                    summary.log(writer_train, session.global_step)

                """Analyse Gradients"""
                if opt['grad_samples'] > 0:
                    print(logging_sep())
                    analyse_gradients_and_log(opt, session.global_step, writer_train, train_logger, loader_train,
                                              model, estimator, parameters, exp_id, tqdm=tqdm)

                """Estimate the ESS"""
                summary_train_ess = evaluation(model, estimator_test_ess, loader_eval_train, exp_id,
                                               device=device, ref_summary=None, max_eval=1000, tqdm=tqdm)

                """Eval on the train set and logging"""
                if opt['only_train_set']:
                    # if the test evaluation is performed on the test set, keep the summary from the evaluation
                    summary_train = summary_train_ess
                else:
                    # seed evaluation such that the subset of `opt['max_eval']` data points remains the same
                    with ManualSeed(seed=opt['seed']):
                        summary_train = evaluation(model, estimator_test, loader_eval_train, exp_id,
                                                   device=device, ref_summary=None, max_eval=opt['max_eval'], tqdm=tqdm)

                    summary_train['loss']['ess'] = summary_train_ess['loss']['ess']

                # log train summary to console and tensorboard
                summary_train.update(parameters_diagnostics)
                log_summary(summary_train, session.global_step, session.epoch, logger=train_logger, best=None,
                            writer=writer_train,
                            exp_id=exp_id)

                """evaluation on the test set and logging"""
                summary_test = evaluation(model, estimator_test, loader_eval_test, exp_id, device=device,
                                          ref_summary=summary_train, max_eval=opt['max_eval'], tqdm=tqdm)

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
                with ManualSeed(seed=opt['seed']):
                    sample_prior_and_save_img("prior-sample", model, logdir, global_step=session.global_step,
                                              writer=writer_test)

                """Checkpointing"""
                session.save()
                print(logging_sep())

        """final testing given the parameters from the best test score"""
        print(f"{logging_sep()}\nFinal validation with best test "
              f"L_{opt['iw_test']} = {session.best_elbo[0]:.3f} at step {session.best_elbo[1]}, epoch = {session.best_elbo[2]}\n{logging_sep()}")

        writer_valid = SummaryWriter(os.path.join(logdir, 'valid'))
        loader_valid = DataLoader(dset_valid, batch_size=opt['test_bs'], shuffle=True, num_workers=1)
        Estimator_valid = VariationalInference
        estimator_valid = Estimator_valid(mc=1, iw=opt['iw_valid'],
                                          sequential_computation=opt['sequential_computation'])

        # load best model and run over the test set
        load_model(model, logdir)
        model.eval()
        agg = Aggregator()
        with torch.no_grad():
            for batch in tqdm(loader_valid, desc=f"[final evaluation] {exp_id}"):
                x, y = preprocess(batch, device)
                diagnostics = test_step(x, model, estimator_valid, y=y)
                agg.update(diagnostics)
            summary = agg.data.to('cpu')

        _, global_step, epoch = session.best_elbo
        log_summary(summary, global_step, epoch, logger=valid_logger, best=None, writer=writer_valid, exp_id=exp_id)

        # prior sampling
        with ManualSeed(seed=opt['seed']):
            sample_prior_and_save_img("prior-sample", model, logdir, global_step=session.global_step,
                                      writer=writer_test)


if __name__ == '__main__':
    run()
