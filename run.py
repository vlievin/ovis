import argparse
import os
from shutil import rmtree

import numpy as np
import torch
from booster import Aggregator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import VAE, Baseline
from lib import VariationalInference, Reinforce, Vimco
from lib import get_shapes_datasets
from lib.gradients import get_gradients_log_total_variance
from lib.logging import sample_model, get_loggers, log_summary, save_model

parser = argparse.ArgumentParser()

# run directory, id and seed
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--exp', default='sandbox', help='experiment directory')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--seed', default=13, type=int, help='random seed')
parser.add_argument('--rm', action='store_true', help='delete previous run')

# epochs, batch size, MC samples, lr
parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--lr_reduce_steps', default=1, type=int, help='number of learning rate reduce steps')

# estimator
parser.add_argument('--estimator', default='reinforce', help='[vi, reinforce, vimco]')
parser.add_argument('--mc_samples', default=4, type=int, help='number of Monte-Carlo samples')
parser.add_argument('--iw_samples', default=1, type=int, help='number of Importance-Weighted samples')
parser.add_argument('--baseline', action='store_true', help='use baseline')

# latent space
parser.add_argument('--N', default=8, type=int, help='number of latent variables')
parser.add_argument('--K', default=8, type=int, help='number of categories for each latent variable')
parser.add_argument('--learn_prior', action='store_true', help='learn the prior')

# model architecture
parser.add_argument('--hdim', default=64, type=int, help='number of hidden units for each layer')
parser.add_argument('--nlayers', default=3, type=int, help='number of hidden layers for the encoder and decoder')
parser.add_argument('--b_nlayers', default=1, type=int, help='number of hidden layers for the baseline')

opt = parser.parse_args()

# defining the run identifier
run_id = f"shapes-vae-{opt.estimator}"
if len(opt.id) > 0:
    run_id += f"-{opt.id}"
run_id += f"-lr{opt.lr:.1E}-bs{opt.bs}-mc{opt.mc_samples}-iw{opt.iw_samples}"
if opt.baseline:
    run_id += f"-baseline{opt.b_nlayers}"
run_id += f"-N{opt.N}-K{opt.K}"
if opt.learn_prior:
    run_id += "-learn-prior"
run_id += f"-arch{opt.hdim}x{opt.nlayers}"

# defining the run directory
logdir = os.path.join(opt.root, opt.exp)
logdir = os.path.join(logdir, run_id)
if os.path.exists(logdir):
    if opt.rm:
        rmtree(logdir)
        os.makedirs(logdir)
else:
    os.makedirs(logdir)

# define logger
base_logger, train_logger, valid_logger = get_loggers(logdir)
base_logger.info(f"Torch version: {torch.__version__}")

# setting the random seed
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

# get datasets
dset_train, dset_valid, dset_test = get_shapes_datasets()
x = dset_train[0]
base_logger.info(f"Dataset size: train = {len(dset_train)}, valid = {len(dset_valid)}")
base_logger.info(f"Sample: x.shape = {x.shape}, x.min = {x.min():.1f}, x.max = {x.max():.1f}, x.dtype = {x.dtype}")

# dataloaders
loader_train = DataLoader(dset_train, batch_size=opt.bs, shuffle=True)
loader_valid = DataLoader(dset_valid, batch_size=2 * opt.bs, shuffle=True)

# define model
torch.manual_seed(opt.seed)
model = VAE(x.shape, opt.N, opt.K, opt.hdim, nlayers=opt.nlayers, learn_prior=opt.learn_prior)

# define baseline
baseline = Baseline(x.shape, opt.b_nlayers, opt.hdim) if opt.baseline else None

# estimator
Estimator = {'vi': VariationalInference, 'reinforce': Reinforce, 'vimco': Vimco}[opt.estimator]
estimator = Estimator(baseline=baseline, mc_samples=opt.mc_samples, iw_samples=opt.iw_samples)

# get device and move models
device = "cuda:0" if torch.cuda.device_count() else "cpu"
model.to(device)
estimator.to(device)
if baseline is not None:
    baseline.to(device)

# optimizer
parameters = list(model.parameters())
if baseline is not None:
    parameters += list(baseline.parameters())
optimizer = torch.optim.Adam(parameters, lr=opt.lr)

# parameters such as beta, freebits, etc..
config = {}

# data aggregator
agg_train = Aggregator()
agg_valid = Aggregator()

# tensorboard writers
writer_train = SummaryWriter(os.path.join(logdir, 'train'))
writer_valid = SummaryWriter(os.path.join(logdir, 'valid'))

# run
best_elbo = (-1e20, 0, 0)
global_step = 0
for epoch in range(1, opt.epochs + 1):

    # sample model
    sample_model("prior-sample", model, logdir, global_step=global_step, writer=writer_valid, seed=opt.seed)

    # train epoch
    optimizer.zero_grad()
    model.train()
    agg_train.initialize()
    for x in tqdm(loader_train):
        x = x.to(device)
        loss, diagnostics, output = estimator(model, x, **config)
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        agg_train.update(diagnostics)
        global_step += 1
    summary_train = agg_train.data.to('cpu')

    # estimate gradients variance
    x = next(iter(loader_train)).to(device)
    summary_train["loss"]["log_grad_var"] = get_gradients_log_total_variance(estimator, model, x, **config)

    # valid epoch
    with torch.no_grad():
        model.eval()
        agg_valid.initialize()
        for x in tqdm(loader_train):
            x = x.to(device)
            _, diagnostics, _ = estimator(model, x, **config)
            agg_valid.update(diagnostics)
        summary_valid = agg_valid.data.to('cpu')

    # update best elbo and save model
    best_elbo = save_model(model, summary_valid, global_step, epoch, best_elbo, logdir)

    # log to console and tensorboard
    log_summary(summary_train, global_step, epoch, logger=train_logger, writer=writer_train)
    log_summary(summary_valid, global_step, epoch, logger=valid_logger, best=best_elbo, writer=writer_valid)

    # reduce learning rate
    lr_freq = (opt.epochs // (opt.lr_reduce_steps + 1))
    if epoch % lr_freq == 0:
        base_logger.info("REDUCING LR")
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2
