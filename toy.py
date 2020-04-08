import argparse
import json
import os
import socket
import traceback
from shutil import rmtree

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lib import ToyVAE
from lib import VariationalInference
from lib.config import get_config
from lib.gradients import get_gradients_statistics
from lib.logging import get_loggers
from lib.utils import notqdm

_sep = os.get_terminal_size().columns * "-"

parser = argparse.ArgumentParser()

# run directory, id and seed
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

# estimator
parser.add_argument('--estimators', default='covbaseline-arithmetic,vimco-arithmetic,pathwise,covbaseline-geometric,vimco-geometric', help='[vi, reinforce, vimco, gs, st-gs]')
parser.add_argument('--iws', default="1000,900,800,700,600,500,400,300,200,100,50,30,20,10,5", help='number of Importance-Weighted samples')
parser.add_argument('--iw_valid', default=1000, type=int, help='number of iw samples for testing')

# noise perturbation for the parameters
parser.add_argument('--noise', default=0.01, type=float, help='scale of the noise added to the optimal parameters')

# evaluation of the gradients
parser.add_argument('--mc_samples', default=1024, type=int, help='number of samples for gradients evaluation')
parser.add_argument('--batch_size', default=2048, type=int, help='batch size used during evaluation')

# dataset
parser.add_argument('--npoints', default=1024, type=int, help='number of datapoints')
parser.add_argument('--D', default=20, type=int, help='number of latent variables')

opt = parser.parse_args()

if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if opt.silent:
    tqdm = notqdm

# defining the run identifier
run_id = f"toy-seed{opt.seed}"
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
    model = ToyVAE((opt.D, ), None, None, None)

    # valid estimator (it is important that all models are evaluated using the same evaluator)
    config_valid = {'tau': 0, 'zgrads': False}
    estimator_valid = VariationalInference(mc=1, iw=opt.iw_valid, sequential_computation=opt.sequential_computation)

    # get device and move models
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    model.to(device)
    estimator_valid.to(device)

    # parse estimators
    estimators = opt.estimators.replace(" ", "").split(",")
    iws = [eval(k) for k in opt.iws.replace(" ", "").split(",")]

    # gradients analysis args
    grad_args = {'seed': opt.seed, 'batch_size': opt.batch_size, 'n_samples': opt.mc_samples,
                 'key_filter': 'qlogits'}


    # generate the dataset
    model.p_mu.data = torch.randn_like(model.p_mu.data)
    x = model.sample_from_prior(N=opt.npoints)['px'].sample()

    print("x:", x.device)

    # evaluate model
    loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    elbo = diagnostics['loss']['elbo'].mean().item()
    base_logger.info(
        f"Before init. | L_{estimator_valid.iw} = {elbo:.3f}")

    # initizalize model using the optimal parameters
    # model.p_mu.data = x.mean(dim=0, keepdim=True).data
    model.q_mu.weight.data = torch.eye(x.shape[1], device=x.device)
    model.q_mu.bias.data = x.mean(dim=0, keepdim=True).data / 2.

    # evaluate model
    loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    elbo = diagnostics['loss']['elbo'].mean().item()
    base_logger.info(
        f"After init. | L_{estimator_valid.iw} = {elbo:.3f}")


    # add perturbation
    model.p_mu.data += opt.noise * torch.randn_like(model.p_mu.data)
    model.q_mu.weight.data += opt.noise * torch.randn_like(model.q_mu.weight.data)
    model.q_mu.bias.data += opt.noise * torch.randn_like(model.q_mu.bias.data)

    loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    elbo = diagnostics['loss']['elbo'].mean().item()
    base_logger.info(
        f"After perturbation | L_{estimator_valid.iw} = {elbo:.3f}")

    # optimization
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # for epoch in range(1, 1000):
    #     model.zero_grad()
    #     loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    #     loss.mean().backward()
    #
    # loss, diagnostics, output = estimator_valid(model, x, backward=False, **config_valid)
    # elbo = diagnostics['loss']['elbo'].mean().item()
    # base_logger.info(
    #     f"After optimization | L_{estimator_valid.iw} = {elbo:.3f}")



    data = []
    for estimator_id in estimators:
        for iw in tqdm(iws, desc=f"{estimator_id} : iws"):
            # create estimator
            Estimator, config = get_config(estimator_id)
            estimator = Estimator(baseline=None, mc=1, iw=iw)
            estimator.to(device)

            # setting the random seed
            torch.manual_seed(opt.seed)

            # evalute variance of the gradients
            grad_data = get_gradients_statistics(estimator, model, x, **grad_args, **config)
            base_logger.info(
                f"{estimator_id}, iw = {iw} | snr = {grad_data.get('snr', 0.):.3E}, variance = {grad_data.get('variance', 0.):.3E}, magnitude = {grad_data.get('magnitude', 0.):.3E}")

            data += [{'seed': opt.seed,
                      'elbo': elbo,
                      'estimator': type(estimator).__name__,
                      'iw': iw,
                      'snr': grad_data.get('snr', 0.).item(),
                      'variance': grad_data.get('variance', 0.).item(),
                      'magnitude': grad_data.get('magnitude', 0.).item()
                      }]

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(logdir, 'data.json'))



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
