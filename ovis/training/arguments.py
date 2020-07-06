import argparse


def add_base_args(parser: argparse.PARSER, exp='sandbox', dataset='shapes'):
    """base experiments arguments use across all exps"""
    parser.add_argument('--dataset', default=dataset,
                        help='dataset [shapes | binmnist | omniglot | fashion | gmm-toy | gaussian-toy]')
    parser.add_argument('--mini', action='store_true',
                        help='use a down-sampled version of the dataset')
    parser.add_argument('--root', default='runs/',
                        help='directory to store training logs')
    parser.add_argument('--data_root', default='data/',
                        help='directory to store the data')
    parser.add_argument('--exp', default=exp,
                        help='experiment directory')
    parser.add_argument('--id', default='', type=str,
                        help='run id suffix')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of dataloader workers')
    parser.add_argument('--rm', action='store_true',
                        help='delete the previous run')
    parser.add_argument('--silent', action='store_true',
                        help='silence tqdm')
    parser.add_argument('--deterministic', action='store_true',
                        help='use deterministic backend')
    parser.add_argument('--sequential_computation', action='store_true',
                        help='compute each iw sample sequentially during validation')


def add_iw_sweep_args(parser: argparse.PARSER, min=5, max=1e2, steps=3):
    """Arguments defining a sweep of the parameter `iw`"""
    parser.add_argument('--iw_min', default=min, type=float,
                        help='min umber of Importance-Weighted samples')
    parser.add_argument('--iw_max', default=max, type=float,
                        help='max number of Importance-Weighted samples')
    parser.add_argument('--iw_steps', default=steps, type=int,
                        help='number of Importance-Weighted samples samples')


def add_active_units_args(parser: argparse.PARSER):
    """Arguments for the `active units` analysis"""
    parser.add_argument('--mc_au_analysis', default=0, type=int,
                        help='number of Monte-Carlo samples used to estimate the number of active units (skip if zero)')
    parser.add_argument('--npoints_au_analysis', default=1000, type=int,
                        help='number of data points x')


def add_gradient_analysis_args(parser: argparse.PARSER):
    """Arguments for the gradient analysis"""
    parser.add_argument('--grad_bs', default=10, type=int,
                        help='gradients analysis batch size')
    parser.add_argument('--grad_samples', default=100, type=int,
                        help='number of MC samples used to evaluate the Variance and SNR of the gradients.')
    parser.add_argument('--grad_key', default='inference_network', type=str,
                        help='key matching the name of the parameters used for the gradients analysis')
    parser.add_argument('--grad_epsilon', default=1e-15, type=float,
                        help='Minimum variance value')


def add_model_architecture_args(parser: argparse.PARSER):
    """Arguments defining the model architecture"""
    parser.add_argument('--model', default='vae',
                        help='[vae, conv-vae, gmm, gaussian-toy, sbm, gaussian-vae]')
    parser.add_argument('--hdim', default=64, type=int,
                        help='number of hidden units for each layer')
    parser.add_argument('--nlayers', default=3, type=int,
                        help='number of hidden layers in each MLP')
    parser.add_argument('--depth', default=3, type=int,
                        help='number of stochastic layers when using hierarchical models')
    parser.add_argument('--b_nlayers', default=1, type=int,
                        help='number of MLP hidden layers for the neural baseline')
    parser.add_argument('--normalization', default='none', type=str,
                        help='normalization layer for the VAE model [none | layernorm | batchnorm]')
    parser.add_argument('--dropout', default=0, type=float,
                        help='dropout value')
    parser.add_argument('--prior', default='normal',
                        help='prior for the VAE model [normal, categorical, bernoulli]')
    parser.add_argument('--N', default=32, type=int,
                        help='number of latent variables for each stochastic layer')
    parser.add_argument('--K', default=8, type=int,
                        help='number of categories when using a categorical prior')
    parser.add_argument('--kdim', default=0, type=int,
                        help='dimension of the keys model for categorical prior')
    parser.add_argument('--learn_prior', action='store_true',
                        help='learn the prior parameters (VAE model)')


def add_run_args(parser: argparse.PARSER):
    """parse the remaining arguments for the script `run.py`"""

    # epochs, batch size, MC samples, lr
    parser.add_argument('--epochs', default=-1, type=int,
                        help='number of epochs (use n_steps if `epochs` < 0)')
    parser.add_argument('--nsteps', default=500000, type=int,
                        help='number of optimization steps')
    parser.add_argument('--optimizer', default='adam',
                        help='[sgd | adam | adamax | rmsprop]')
    parser.add_argument('--lr', default=2e-3, type=float,
                        help='base learning rate')
    parser.add_argument('--freebits', default=None, type=float,
                        help='number of `freebits` per layer')
    parser.add_argument('--baseline_lr', default=5e-3, type=float,
                        help='learning rate for the weight of the baseline')
    parser.add_argument('--bs', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--valid_bs', default=50, type=int,
                        help='validation evaluation batch size')
    parser.add_argument('--test_bs', default=50, type=int,
                        help='test evaluation batch size')
    parser.add_argument('--lr_reduce_steps', default=0, type=int,
                        help='number of learning rate reduce steps')
    parser.add_argument('--only_train_set', action='store_true',
                        help='only use the training dataset: useful to isolate the optimization behaviour.')

    # estimator & warmup
    parser.add_argument('--estimator', default='reinforce',
                        help='see estimators.config for the full list [pathwise-iwae, reinforce, reinforce-baseline, '
                             'ovis-S*, ovis-gamma*, tvo, vimco-arithmetic]')
    parser.add_argument('--mc', default=1, type=int,
                        help='number of `outer` Monte-Carlo samples')
    parser.add_argument('--iw', default=1, type=int,
                        help='number of `inner` Importance-Weighted samples')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='Beta weight for the KL term (i.e. Beta-VAE)')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha weight for the unormalized weights: w_k^alpha')
    parser.add_argument('--warmup', default=0, type=int,
                        help='period of the posterior warmup (alpha : `alpha_init` -> 0)')
    parser.add_argument('--warmup_offset', default=0, type=int,
                        help='number of initial steps before increasing beta')
    parser.add_argument('--warmup_mode', default='log', type=str,
                        help='interpolation mode [linear | log]')
    parser.add_argument('--alpha_init', default=0.9, type=float,
                        help='initial alpha value')

    # evaluation
    parser.add_argument('--eval_freq', default=10, type=int,
                        help='frequency of evalution [evaluate log p(x), analyse grads and sample the prior]')
    parser.add_argument('--max_eval', default=None, type=int,
                        help='maximum number of data points for evaluation')
    parser.add_argument('--iw_valid', default=100, type=int,
                        help='number of Importance-Weighted samples for validation')
    parser.add_argument('--iw_test', default=1000, type=int, help='number of Importance-Weighted samples for testing')
