import torch
from torch.optim import SGD, Adam, Adamax, RMSprop

from ovis import VAE, GaussianToyVAE, GaussianMixture, SigmoidBeliefNetwork, GaussianVAE, Baseline, VariationalInference
from ovis.estimators.config import get_config
from .utils import get_dataset_mean


def init_model(opt, x, loader=None):
    # compute mean value of train set
    xmean = None if loader is None else get_dataset_mean(loader)

    # define the hyperparameters
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
        'x_mean': xmean
    }

    # change the model id based on the chosen dataset if required
    model_id = {'gmm-toy': 'gmm-toy',
                'gaussian-toy': 'gaussian-toy',
                'air': 'air'}.get(opt.dataset, opt.model)

    # get the right constructor
    _MODEL = {'vae': VAE,
              'gaussian-toy': GaussianToyVAE,
              'gmm-toy': GaussianMixture,
              'sbm': SigmoidBeliefNetwork,
              'gaussian': GaussianVAE}[model_id]

    # init the model
    torch.manual_seed(opt.seed)
    model = _MODEL(**hyperparams)
    return model, hyperparams


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


def init_test_estimator(opt):
    """
    Initialize the 2 validation estimators.
    * a. estimator_valid: used to compute L_K_valid
    * b. estimator_valid_as_training: used to compute L_K_train

    The two estimators are used toe estimate KL(q||p) = log \hat{p(x)} - L_K_train considering log \hat{p(x)} = L_K_valid
    """
    Estimator = VariationalInference
    config = {'tau': 0, 'zgrads': False}

    # the main test estimator with K = opt.iw_test
    estimator = Estimator(mc=1, iw=opt.iw_test, sequential_computation=opt.sequential_computation)
    estimator_ess = Estimator(mc=1, iw=opt.iw, sequential_computation=opt.sequential_computation)

    return estimator, estimator_ess, config


def init_optimizers(opt, model, estimator):
    """Initialize estimators both for the model's parameters and the baselines/estimator's parameters"""
    optimizers = []
    _OPT = {'sgd': SGD, 'adam': Adam, 'adamax': Adamax, 'rmsprop': RMSprop}[opt.optimizer]
    optimizers += [_OPT(model.parameters(), lr=opt.lr)]
    if len(list(estimator.parameters())):
        optimizers += [torch.optim.Adam(estimator.parameters(), lr=opt.baseline_lr)]

    return optimizers
