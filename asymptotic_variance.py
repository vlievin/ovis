import argparse
import json

import pandas as pd
import torch
from booster.utils import logging_sep, available_device

from ovis.analysis.gradients import get_gradients_statistics
from ovis.estimators.config import get_config
from ovis.models import GaussianToyVAE
from ovis.reporting.asymptotic import *
from ovis.reporting.parsing import format_estimator_name
from ovis.reporting.style import *
from ovis.training.evaluation import evaluate_minibatch_and_log
from ovis.training.initialization import init_logging_directory
from ovis.training.logging import get_loggers, log_grads_data
from ovis.training.utils import get_hash_from_opt
from ovis.utils.success import Success
from ovis.utils.utils import notqdm, ManualSeed


def init_estimator(estimator_id, iw):
    """initialize the gradient estimator based on the `estimator_id` and the number of particles `iw`"""
    Estimator, config = get_config(estimator_id)
    return Estimator(baseline=None, mc=1, iw=iw, **config), config


parser = argparse.ArgumentParser()

# run directory, id and seed
parser.add_argument('--root', default='runs/',
                    help='directory to store training logs')
parser.add_argument('--data_root', default='data/',
                    help='directory to store the data')
parser.add_argument('--exp', default='asymptotic-variance-final',
                    help='experiment directory')
parser.add_argument('--id', default='', type=str,
                    help='run id suffix')
parser.add_argument('--seed', default=1, type=int,
                    help='random seed')
parser.add_argument('--rm', action='store_true',
                    help='delete previous run')
parser.add_argument('--silent', action='store_true',
                    help='silence tqdm')
parser.add_argument('--deterministic', action='store_true',
                    help='use deterministic backend')

# estimator, perturbation level and number of particles
parser.add_argument('--estimators', default='ovis-gamma0, pathwise-iwae',
                    help='accepts comma separated list')
parser.add_argument('--epsilon', default='0.01', type=str,
                    help='scale of the noise added to the optimal parameters [accepts comma separated list]')
parser.add_argument('--iw_min', default=5, type=float,
                    help='min umber of Importance-Weighted samples')
parser.add_argument('--iw_max', default=1e2, type=float,
                    help='max number of Importance-Weighted samples')
parser.add_argument('--iw_steps', default=3, type=int,
                    help='number of Importance-Weighted samples samples')
parser.add_argument('--iw_valid', default=1000, type=int,
                    help='number of iw samples for testing')

# evaluation of the gradients
parser.add_argument('--key_filter', default='b', type=str,
                    help='identifiant of the parameters/tensor for the gradients analysis')
parser.add_argument('--mc_samples', default=300, type=float,
                    help='number of Monte-Carlo samples used for gradients evaluations')
parser.add_argument('--samples_per_batch', default=80000, type=int,
                    help='number of samples per batch [N = bs x ms x iw].')
parser.add_argument('--draw_individual', action='store_true',
                    help='draw SNR and Variance independently for each parameter.')

# dataset
parser.add_argument('--npoints', default=1024, type=int,
                    help='number of datapoints')
parser.add_argument('--D', default=20, type=int,
                    help='latent space dimension')

# geometric spacing of the particles `iws`
opt = parser.parse_args()
iws = [int(k) for k in np.geomspace(start=opt.iw_min, stop=opt.iw_max, num=opt.iw_steps)[::-1]]

if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if opt.silent:
    tqdm = notqdm

# defining the run identifier
deterministic_id = get_hash_from_opt(opt)
run_id = f"asymptotic-{opt.estimators}-iw{opt.iw_min}-{opt.iw_max}-{opt.iw_steps}-seed{opt.seed}-eps{opt.epsilon}"
if opt.exp != "":
    run_id += f"-{opt.exp}"
run_id += f"{deterministic_id}"
_exp_id = f"asymptotic-{opt.exp}-{opt.seed}"

# defining the run directory
logdir = init_logging_directory(opt, run_id)

# save configuration
with open(os.path.join(logdir, 'config.json'), 'w') as fp:
    _opt = vars(opt)
    fp.write(json.dumps(_opt, default=lambda x: str(x), indent=4))

# wrap the training loop inside with `Success` to write the outcome of the run to a file
with Success(logdir=logdir):
    # define logger
    print(logging_sep("="))
    base_logger, *_ = get_loggers(logdir, keys=['base'])
    base_logger.info(f"Run id: {run_id}")
    base_logger.info(f"Logging directory: {logdir}")
    base_logger.info(f"Torch version: {torch.__version__}")

    # setting the random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # define model
    torch.manual_seed(opt.seed)
    model = GaussianToyVAE(xdim=(opt.D,), npoints=opt.npoints)

    # valid estimator (it is important that all models are evaluated using the same evaluator)
    Estimator, config_ref = get_config("pathwise-iwae")
    estimator_ref = Estimator(mc=1, iw=opt.iw_valid, **config_ref)

    # get device and move models
    device = available_device()
    model.to(device)
    estimator_ref.to(device)

    # parse estimators
    estimators = opt.estimators.replace(" ", "").split(",")

    # get the dataset
    x = model.dset

    # evaluate model at initialization
    with ManualSeed(seed=opt.seed):
        diagnostics = evaluate_minibatch_and_log(estimator_ref, model, x, config_ref, base_logger,
                                                 "Random Initialisation")

    grads_stats = []
    grads_data = []
    epsilons = [eval(x) for x in opt.epsilon.split(",")]
    global_grad_args = {'seed': opt.seed,
                        'samples_per_batch': opt.samples_per_batch,
                        'key_filter': opt.key_filter}

    for epsilon in epsilons:

        # initialize the model using the optimal parameters
        model.set_optimal_parameters()

        # evaluate the model
        with ManualSeed(seed=opt.seed):
            diagnostics = evaluate_minibatch_and_log(estimator_ref, model, x, config_ref, base_logger,
                                                     "Optimal parameters")

        # add perturbation to the weights
        model.perturbate_weights(epsilon)

        # evaluate model
        with ManualSeed(seed=opt.seed):
            diagnostics = evaluate_minibatch_and_log(estimator_ref, model, x, config_ref, base_logger,
                                                     "After perturbation")

        # define the gradients analysis arguments and the meta-data
        meta = {'seed': opt.seed, 'noise': epsilon, 'mc_samples': int(opt.mc_samples),
                **{k: v.mean().item() for k, v in diagnostics['loss'].items()}}
        grad_args = {'mc_samples': int(opt.mc_samples), **global_grad_args}
        idx = None

        for estimator_id in estimators:
            print(logging_sep())
            for iw in iws:
                base_logger.info(f"{estimator_id} [K = {iw}]")

                # create estimator
                estimator, config = init_estimator(estimator_id, iw)
                estimator.to(device)

                # evalute variance of the gradients
                with ManualSeed(seed=opt.seed):
                    analysis_data, grads_meta = get_gradients_statistics(estimator, model, x,
                                                                         return_grads=True, **grad_args, **config)

                # log grads info
                log_grads_data(analysis_data, base_logger, estimator_id, iw)

                # store results
                identifier = {'estimator': estimator_id, 'iw': iw, **meta}

                # get statistics for each parameter separately
                individual_stats = {f"{k}-{i}": v_i for k, v in grads_meta.items() for i, v_i in enumerate(v) if
                                    k in ['magnitude', 'var', 'snr']}
                grads_stats += [{
                    **{f"grads-{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in
                       analysis_data['grads'].items()},
                    **{f"individual-{k}": v.item() for k, v in individual_stats.items()},
                    **{f"snr-{k}": v.item() for k, v in analysis_data['snr'].items()},
                    **identifier
                }]

                # store grads
                grads = grads_meta.get('grads')
                grads = grads.view(-1, grads.shape[-1]).transpose(1, 0)

                # get the index of the expected gradient sorted by abs. value
                if idx is None:
                    _, idx = grads.mean(dim=1).abs().sort(descending=True)

                # sort gradients according to idx. Identical results are obtained without sorting however sorting
                # ensures tracking a parameter with a non-trivial gradient
                grads = grads[idx]

                # return gradients for the first param
                for g in grads[0, :]:
                    grads_data += [
                        {'param': 'all', 'grad': g.item(), **identifier}]

    # convert into DataFrames
    df = pd.DataFrame(grads_stats)
    grads_data = pd.DataFrame(grads_data)

    # Save to CSV
    df.to_csv(os.path.join(logdir, 'data.csv'))
    grads_data.to_csv(os.path.join(logdir, 'grads.csv'))

    # plotting
    set_matplotlib_style()
    df['estimator'] = list(map(format_estimator_name, df['estimator'].values))
    plot_statistics(df, opt, logdir)
    if len(grads_data):
        grads_data['estimator'] = list(map(format_estimator_name, grads_data['estimator'].values))
        plot_gradients_distribution(grads_data, logdir)
