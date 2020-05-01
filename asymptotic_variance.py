import argparse
import json
import socket
import traceback
from shutil import rmtree

import pandas as pd

from lib.logging import get_loggers
from lib.models import GaussianToyVAE
from lib.utils import notqdm
from lib.variance_utils import *

colors = sns.color_palette()
_sep = os.get_terminal_size().columns * "-"

parser = argparse.ArgumentParser()

# run directory, id and seed
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--data_root', default='data/', help='directory to store the data')
parser.add_argument('--exp', default='asymptotic-variance-0.1', help='experiment directory')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--seed', default=13, type=int, help='random seed')
parser.add_argument('--workers', default=1, type=int, help='dataloader workers')
parser.add_argument('--rm', action='store_true', help='delete previous run')
parser.add_argument('--silent', action='store_true', help='silence tqdm')
parser.add_argument('--deterministic', action='store_true', help='use deterministic backend')
parser.add_argument('--sequential_computation', action='store_true',
                    help='compute each iw sample sequential during validation')

# estimator
parser.add_argument('--estimators',
                    default='copt-arithmetic,vimco-arithmetic,pathwise',
                    help='[vi, reinforce, vimco, gs, st-gs]')

parser.add_argument('--iw_min', default=5, type=float, help='min umber of Importance-Weighted samples')
parser.add_argument('--iw_max', default=1e3, type=float, help='max number of Importance-Weighted samples')
parser.add_argument('--iw_steps', default=5, type=float, help='number of Importance-Weighted samples samples')
parser.add_argument('--iw_valid', default=1000, type=int, help='number of iw samples for testing')

parser.add_argument('--oracle', default='pathwise-iwae', type=str, help='oracle estimator id')
parser.add_argument('--iw_oracle', default=1000, type=int, help='number of iw samples to find the true gradients')
parser.add_argument('--mc_oracle', default=10000, type=int,
                    help='number of mc samples used to compute the estimae of the oracle gradients')

# noise perturbation for the parameters
parser.add_argument('--noise', default='0.01', type=str, help='scale of the noise added to the optimal parameters')

# evaluation of the gradients
parser.add_argument('--key_filter', default='tensor:b', type=str,
                    help='identifiant of the parameters/tensor for the gradients analysis')
parser.add_argument('--mc_samples', default=100, type=int, help='number of samples for gradients evaluation')
parser.add_argument('--max_points', default=0, type=int, help='number of data points to evaluate the grads in')
parser.add_argument('--batch_size', default=80000, type=int,
                    help='number of samples per batch size used during evaluation')
parser.add_argument('--grads_dist', action='store_true', help='plot distributions of gradients for each parameter')
parser.add_argument('--use_all_params', action='store_true', help='look a the aggregated dist of grads')
parser.add_argument('--batch_grads', action='store_true',
                    help='analyse batch gradients instead of the gradients for each individual x')
parser.add_argument('--draw_individual', action='store_true', help='draw statistics independently for each parameter')

# dataset
parser.add_argument('--npoints', default=1024, type=int, help='number of datapoints')
parser.add_argument('--D', default=20, type=int, help='number of latent variables')

opt = parser.parse_args()

iws = [int(k) for k in np.geomspace(start=3, stop=opt.iw_max, num=opt.iw_steps)[::-1]]

if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if opt.silent:
    tqdm = notqdm

# defining the run identifier
run_id = f"toy-{opt.estimators}-iw{opt.iw_min}-{opt.iw_max}-{opt.iw_steps}-oracle={opt.oracle}-seed{opt.seed}-noise{opt.noise}-mc{opt.mc_samples}-key{opt.key_filter}-pts{opt.npoints}-D{opt.D}"
if opt.id != "":
    run_id += f"-{opt.id}"
if opt.batch_grads:
    run_id += f"-batch_grads"
if opt.use_all_params:
    run_id += f"-allp"
_exp_id = f"toy-{opt.exp}-{opt.seed}"

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

try:
    # define logger
    base_logger, train_logger, valid_logger, test_logger = get_loggers(logdir)
    base_logger.info(f"Torch version: {torch.__version__}")

    # setting the random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # define model
    torch.manual_seed(opt.seed)
    model = GaussianToyVAE(xdim=(opt.D,))

    # valid estimator (it is important that all models are evaluated using the same evaluator)
    Estimator, config_ref = get_config("pathwise-iwae")
    estimator_ref = Estimator(mc=1, iw=opt.iw_valid)
    Estimator, config_oracle = get_config(opt.oracle)
    oracle = Estimator(mc=1, iw=opt.iw_oracle)

    # get device and move models
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    model.to(device)
    estimator_ref.to(device)

    # parse estimators
    estimators = opt.estimators.replace(" ", "").split(",")

    # generate the dataset
    x = gen_dataset(model, opt.npoints)

    # evaluate model
    diagnostics = evaluate(estimator_ref, model, x, config_ref, opt.seed, base_logger, "Before perturbation")

    data = []
    grads = []
    noises = [eval(x) for x in opt.noise.split(",")]
    global_grad_args = {'seed': opt.seed, 'batch_size': opt.batch_size, 'key_filter': opt.key_filter}
    for noise in noises:
        print(_sep)
        base_logger.info(f">> Noise = {noise}")

        # initizalize model using the optimal parameters
        set_optimal_parameters(model, x)

        # evaluate model
        diagnostics = evaluate(estimator_ref, model, x, config_ref, opt.seed, base_logger, "After init.")

        # add perturbation to the weights
        perturbate_weights(model, noise)

        # evaluate model
        diagnostics = evaluate(estimator_ref, model, x, config_ref, opt.seed, base_logger, "After perturbation")

        # compute the true direction of the gradients
        true_grads = compute_true_grads(oracle, model, x, opt.mc_oracle, **global_grad_args, **config_oracle)
        # order gradients by magnitude
        _, grads_idx = true_grads.abs().sort(descending=True)

        # gradients analysis args and config
        meta = {'seed': opt.seed, 'noise': noise, 'mc_samples': opt.mc_samples,
                **{k: v.mean().item() for k, v in diagnostics['loss'].items()}}
        grad_args = {'n_samples': opt.mc_samples, 'true_grads': true_grads, **global_grad_args}

        for estimator_id in estimators:
            print(_sep)
            base_logger.info(f">> Analysis for {estimator_id}")
            for iw in iws:
                base_logger.info(f">> K = {iw}")

                # create estimator
                estimator, config = get_estimator(estimator_id, iw)
                estimator.to(device)

                # evalute variance of the gradients
                analysis_data, grads_meta = get_batch_gradients_statistics(estimator, model, x,
                                                                           use_batch_grads=opt.batch_grads,
                                                                           return_grads=opt.grads_dist,
                                                                           use_dsnr=True, **grad_args, **config)

                # log grads info
                log_grads_data(analysis_data, base_logger, estimator_id, iw)

                # store results
                # get statistics for each parameter separately
                individual_stats = {f"{k}-{i}": v_i for k, v in grads_meta.items() for i, v_i in enumerate(v) if
                                    k in ['magnitude', 'var', 'snr']}
                data += [{
                    'estimator': estimator_id,
                    'iw': iw,
                    **{f"grads-{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in
                       analysis_data['grads'].items()},
                    **{f"individual-{k}": v.item() for k, v in individual_stats.items()},
                    **{f"snr-{k}": v.item() for k, v in analysis_data['snr'].items()},
                    **meta
                }]

                # store grads
                if opt.grads_dist:
                    grads_ = grads_meta.get('grads')
                    grads_ = grads_.view(-1, grads_.shape[-1]).transpose(1, 0)

                    # reindex by `true_grads` magnitude so all grads expectations are positives
                    grads_ = grads_[grads_idx]

                    # get only positive biases using the sign of the true_grads as a reference.
                    if opt.use_all_params:
                        u = true_grads[None]
                        dir = 2 * (u < 0).float() - 1.
                        grads_ = grads_ * dir

                    if opt.use_all_params:
                        # return gradients for all parameters
                        for g in grads_.view(-1):
                            grads += [
                                {'noise': noise, 'param': 'all', 'estimator': estimator_id, 'iw': iw, 'grad': g.item()}]
                    else:
                        # return gradients for the first param
                        for g in grads_[0, :]:
                            grads += [
                                {'noise': noise, 'param': 'all', 'estimator': estimator_id, 'iw': iw, 'grad': g.item()}]

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(logdir, 'data.csv'))
    base_logger.info(f"# path = {os.path.abspath(logdir)}")

    plot_statistics(df, opt, logdir)

    if len(grads):
        grads = pd.DataFrame(grads)
        plot_gradients_distribution(grads, logdir)


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
