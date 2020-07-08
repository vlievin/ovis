import os
from shutil import rmtree
from typing import *

import torch
from torch.optim import SGD, Adam, Adamax, RMSprop

from ovis import TemplateModel, VAE, ConvVAE, GaussianToyVAE, GaussianMixture, SigmoidBeliefNetwork, GaussianVAE, Baseline, \
    VariationalInference
from ovis.estimators import GradientEstimator
from ovis.estimators.config import parse_estimator_id
from .utils import get_dataset_mean


def init_model(opt, x, loader=None) -> Tuple[TemplateModel, Dict]:
    hyperparams = {
        **{k: opt[k] for k in
           ['N', 'K', 'hdim', 'kdim', 'nlayers', 'depth', 'learn_prior', 'prior', 'normalization', 'dropout']},
        'xdim': x.shape,
        'x_mean': None if loader is None else get_dataset_mean(loader),
    }

    # change the model id based on the chosen dataset if required
    model_id = {'gmm': 'gmm',
                'gaussian-toy': 'gaussian-toy',
                'air': 'air'}.get(opt['dataset'], opt['model'])

    # get the right constructor
    MODEL = {'vae': VAE,  # VAE parameterized by MLPs
             'conv-vae': ConvVAE,  # VAE parameterized by convolutions
             'gaussian-toy': GaussianToyVAE,  # Gaussian model for the asymptotic variance study
             'gmm': GaussianMixture,  # Gaussian-Mixture-Model
             'sbm': SigmoidBeliefNetwork,  # official TVO SBM model
             'gaussian-vae': GaussianVAE}[model_id]  # official TVO gaussian VAE model

    # init the model
    torch.manual_seed(opt['seed'])
    model = MODEL(**hyperparams)
    return model, hyperparams


def init_neural_baseline(opt, x) -> Baseline:
    """define the neural baseline"""
    return Baseline(x.shape, opt['b_nlayers'], opt['hdim'])


def init_main_estimator(opt, baseline=None, iw=None, mc=None) -> GradientEstimator:
    """initialize the training estimator and its configuration dict"""
    Estimator_cls, config = parse_estimator_id(opt['estimator'])
    estimator = Estimator_cls(baseline=baseline,
                              mc=opt['mc'] if mc is None else mc,
                              iw=opt['iw'] if iw is None else iw,
                              **config)

    return estimator


def init_test_estimator(opt) -> Tuple[GradientEstimator, GradientEstimator]:
    """
    Initialize the 2 validation estimators.
    * a. estimator_valid: used to compute L_K_valid
    * b. estimator_valid_as_training: used to compute L_K_train

    The two estimators are used toe estimate KL(q||p) = log \hat{p(x)} - L_K_train considering log \hat{p(x)} = L_K_valid
    """

    # the main test estimator with `K = opt.iw_test`
    estimator = VariationalInference(mc=1, iw=opt['iw_test'], sequential_computation=opt['sequential_computation'])
    # test estimator with `K = opt.iw` to estimate the training ESS
    estimator_ess = VariationalInference(mc=1, iw=opt['iw'], sequential_computation=opt['sequential_computation'])

    return estimator, estimator_ess


def init_optimizers(opt, model, estimator) -> List[torch.optim.Optimizer]:
    """Initialize estimators both for the model's parameters and the baselines/estimator's parameters"""
    optimizers = []
    _OPT = {'sgd': SGD, 'adam': Adam, 'adamax': Adamax, 'rmsprop': RMSprop}[opt['optimizer']]
    optimizers += [_OPT(model.parameters(), lr=opt['lr'])]
    if len(list(estimator.parameters())):
        optimizers += [torch.optim.Adam(estimator.parameters(), lr=opt['baseline_lr'])]

    return optimizers


def init_logging_directory(opt, run_id) -> str:
    """initialize the directory where will be saved the config, model's parameters and tensorboard logs"""
    logdir = os.path.join(opt['root'], opt['exp'])
    logdir = os.path.join(logdir, run_id)
    if os.path.exists(logdir):
        if opt['rf']:
            rmtree(logdir)
            os.makedirs(logdir)
    else:
        os.makedirs(logdir)

    return logdir
