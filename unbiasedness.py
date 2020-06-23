import argparse
import json
import socket
import traceback
from shutil import rmtree

import pandas as pd
from matplotlib.lines import Line2D

from lib.logging import get_loggers
from lib.models import GaussianToyVAE
from lib.utils import notqdm
from lib.variance_plotting import *

colors = sns.color_palette()
_sep = os.get_terminal_size().columns * "-"

# commands
#  python unbiasedness.py --mc_samples 100 --D 2 --npoints 3 --mc_samples 100000 --iw 20 --noise 10.0

parser = argparse.ArgumentParser()

# run directory, id and seed
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--data_root', default='data/', help='directory to store the data')
parser.add_argument('--exp', default='ubiasedness-0.1', help='experiment directory')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--seed', default=13, type=int, help='random seed')
parser.add_argument('--workers', default=1, type=int, help='dataloader workers')
parser.add_argument('--rm', action='store_true', help='delete previous run')
parser.add_argument('--silent', action='store_true', help='silence tqdm')
parser.add_argument('--deterministic', action='store_true', help='use deterministic backend')
parser.add_argument('--sequential_computation', action='store_true',
                    help='compute each iw sample sequential during validation')

parser.add_argument('--iw', default=50, type=float, help='min umber of Importance-Weighted samples')
parser.add_argument('--iw_valid', default=50, type=float, help='min umber of Importance-Weighted samples')

# noise perturbation for the parameters
parser.add_argument('--noise', default='0.01', type=str, help='scale of the noise added to the optimal parameters')

# evaluation of the gradients
parser.add_argument('--mc_samples', default=1000, type=int, help='number of samples for gradients evaluation')

# dataset
parser.add_argument('--npoints', default=100, type=int, help='number of datapoints')
parser.add_argument('--D', default=20, type=int, help='number of latent variables')

opt = parser.parse_args()


def running_avg_norm(data, spacing=100, offset=0, min_step=50):
    # running average
    idx = []
    serie = []
    _sum = None
    _count = 0
    for i, x in enumerate(data):
        _count += 1
        if _sum is None:
            _sum = x
        else:
            _sum += x

        if i >= min_step and (i == min_step or (offset + i) % spacing == 0):
            serie += [np.sqrt(np.sum((_sum / _count) ** 2))]
            idx += [_count]

    return idx, serie


def compute_dlogits(output):
    z, qz = [output[k] for k in ['z', 'qz']]

    assert len(qz) == 1
    z, qz = z[0], qz[0]

    # compute log probs
    qlogits = qz.logits
    log_qz = qz.log_prob(z)

    # d q(z|x) / d qlogits
    d_qlogits, = torch.autograd.grad(
        [log_qz], [qlogits], grad_outputs=torch.ones_like(log_qz), retain_graph=True, allow_unused=True)

    # reshaping d_qlogits and qlogits
    N, K = d_qlogits.size()[1:] if len(d_qlogits.shape) > 2 else (d_qlogits.size(1), 1)

    return d_qlogits.view(-1, N, K).detach()


def get_grads_and_controls(L_k, log_wk, v_k, log_Znok, hk, mode):
    K = log_wk.size(2)
    log_Znok = log_Znok.squeeze(-1)
    # compute c_k
    if mode == 'reinforce':
        c_k = torch.zeros_like(log_Znok)
    elif mode == 'vimco':
        c_k = log_Znok
    elif mode == 'copt-ess':
        _max, idx = log_wk.max(dim=2, keepdim=True)
        mask = (log_wk == _max).float()
        # c_k = log Z-k - 1_{k = argmax w_k}
        c_k = log_Znok + np.log(K - 1) - np.log(K) - mask
    elif mode == 'copt-biased':
        c_k = log_Znok + np.log(K - 1) - np.log(K) - v_k
    elif mode == 'copt':
        c_k = log_Znok + np.log(K - 1) - np.log(K)
    elif mode == 'ww':
        c_k = L_k.unsqueeze(-1) - 2 * v_k

    # compute grads and control variate
    grads = torch.sum((L_k.unsqueeze(-1) - v_k - c_k).unsqueeze(-1) * hk, 2).mean(1)
    controls = torch.sum((- c_k).unsqueeze(-1) * hk, 2).mean(1)

    return {'grad': grads, 'control': controls}


def compute_grads(estimator, model, x):
    bs = x.size(0)
    output = estimator.evaluate_model(model, x)
    log_px_z, log_pz, log_qz = [output[k] for k in ('log_px_z', 'log_pz', 'log_qz')]
    iw_data = estimator.compute_iw_bound(log_px_z, log_pz, log_qz, detach_qlogits=True, beta=1)
    L_k, ess, log_wk = [iw_data[k] for k in ('L_k', 'ess', 'log_wk')]
    log_wk = log_wk.view(-1, estimator.mc, estimator.iw)

    # compute d\dphi log q(z|x)
    hk = compute_dlogits(output).view(bs, estimator.mc, estimator.iw, -1)

    with torch.no_grad():
        # compute log Z^{-k}
        log_Znok, *_ = Vimco.compute_control_variate(estimator, x, arithmetic=True, **iw_data)

        # compute v_k
        v_k = log_wk.softmax(2)

        return {mode: get_grads_and_controls(L_k, log_wk, v_k, log_Znok, hk, mode) for mode in
                ['reinforce', 'vimco', 'copt', 'copt-ess', 'copt-biased', 'ww']}


if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if opt.silent:
    tqdm = notqdm

# defining the run identifier
run_id = f"uniasedness-iw{opt.iw}-seed{opt.seed}-noise{opt.noise}-mc{opt.mc_samples}-pts{opt.npoints}-D{opt.D}"
if opt.id != "":
    run_id += f"-{opt.id}"
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
    model = GaussianToyVAE(xdim=(opt.D,), npoints=opt.npoints)

    # valid estimator (it is important that all models are evaluated using the same evaluator)
    Estimator, config_ref = get_config("vimco")
    estimator_ref = Estimator(mc=1, iw=opt.iw_valid)

    # get device and move models
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    model.to(device)
    estimator_ref.to(device)

    # get the dataset
    x = model.dset

    # evaluate model
    diagnostics = evaluate(estimator_ref, model, x, config_ref, opt.seed, base_logger, "Before perturbation")

    data = []
    grads = {}
    controls = {}
    noises = [eval(x) for x in opt.noise.split(",")]
    global_grad_args = {'seed': opt.seed}
    for noise in noises:
        print(_sep)
        base_logger.info(f">> Noise = {noise}")

        # initizalize model using the optimal parameters
        model.set_optimal_parameters()

        # evaluate model
        diagnostics = evaluate(estimator_ref, model, x, config_ref, opt.seed, base_logger, "After init.")

        # add perturbation to the weights
        model.perturbate_weights(noise)

        # evaluate model
        diagnostics = evaluate(estimator_ref, model, x, config_ref, opt.seed, base_logger, "After perturbation")

        # gradients analysis args and config
        meta = {'seed': opt.seed, 'noise': noise, 'mc_samples': opt.mc_samples,
                **{k: v.mean().item() for k, v in diagnostics['loss'].items()}}
        grad_args = {'n_samples': opt.mc_samples, **global_grad_args}

        for iw in [opt.iw]:

            # evalute variance of the gradients
            x_ = x[None].repeat(opt.mc_samples, *(1 for _ in x.shape)).view(-1, *x.shape[1:])
            outputs = compute_grads(estimator_ref, model, x_)
            for estimator_id, output in outputs.items():
                grad = output['grad']
                control = output['control']

                # take the mini-batch gradients
                grad = grad.view(opt.mc_samples, x.shape[0], -1).mean(1)
                control = control.view(opt.mc_samples, x.shape[0], -1).mean(1)

                # store results
                data += [{
                    'estimator': estimator_id,
                    'iw': iw,
                    'estimate': grad.mean(0).data.numpy(),
                    'magnitude': grad.mean(0).norm(p=2).abs().item(),
                    'variance': grad.var(0).mean().item(),
                    'snr': (grad.mean(0).abs() / (eps + grad.std(0))).mean().item()
                }]

                # store grads
                controls[estimator_id] = control.data.numpy()
                grads[estimator_id] = grad.data.numpy()

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(os.path.join(logdir, 'data.csv'))
    base_logger.info(f"# path = {os.path.abspath(logdir)}")

    print(_sep)
    print("Summary:")
    print(_sep)
    print(df)

    # correlation matrix
    from sklearn.metrics.pairwise import cosine_similarity

    ids = df['estimator'].values
    gs = np.concatenate([x[None, :] for x in df['estimate'].values], axis=0)
    gs = gs / np.linalg.norm(gs, axis=1, keepdims=True)

    c = gs @ gs.T

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.imshow(c)

    ax.set_xticks(np.arange(0, c.shape[0], c.shape[0] * 1.0 / len(ids)))
    # ax.set_yticks(np.arange(0, c.shape[1], c.shape[1] * 1.0 / len(ids)))

    ax.set_xticklabels(ids)
    # ax.set_yticklabels(ids)
    plt.colorbar()
    plt.title("Correlation matrix for $\mathbb{E}[g]$")
    plt.savefig(os.path.join(logdir, 'corr.png'))
    plt.close()

    # convergence plot
    plt.figure(figsize=(12, 8))
    patches = []
    labels = []
    for k, estimator_id in enumerate(df['estimator'].unique()):
        color = sns.color_palette()[k]
        _grads = running_avg_norm(grads[estimator_id], offset=k, min_step=50)
        _controls = running_avg_norm(controls[estimator_id], offset=k, min_step=50)
        plt.loglog(*_grads, linestyle='-', color=color, alpha=0.8)
        plt.loglog(*_controls, linestyle=':', color=color, alpha=0.8)

        # legend
        patches += [Line2D([0], [0], color=color, lw=2)]
        labels += [estimator_id]

    # linestyles
    patches += [Line2D([0], [0], color="gray", lw=2, linestyle="-"),
                Line2D([0], [0], color="gray", lw=2, linestyle=":")]
    labels += ["grad", "control"]

    plt.title(
        f"Convergence of the gradients and the control variates (K = {opt.iw}, ESS = {diagnostics['loss']['ess'].mean().item():.3f})")
    plt.ylabel(r"$\| \cdot \|$")
    plt.xlabel(r"MC samples")
    plt.legend(patches, labels)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'convergence.png'))
    plt.close()









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
