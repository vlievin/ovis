import argparse
import json
import os
import traceback

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotmap import DotMap

sns.set()

try:
    from tbparser.summary_reader import SummaryReader
    from tbparser import EventsFileReader
except:
    print("You probably need to install tbparser:\n   pip install git+https://github.com/velikodniy/tbparser.git")
    exit()

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='experiment directory')
parser.add_argument('--output', default='reports/', help='output directory')
parser.add_argument('--exp', default='exclude-sample-0.1', type=str, help='experiment id')
parser.add_argument('--metric', default='loss/elbo', type=str, help='metric to track')
parser.add_argument('--latex', action='store_true', help='print as latex table')
parser.add_argument('--float_format', default=".3f", help='float format')
parser.add_argument('--nsamples', default=64, type=int, help='number of points in the line plot')
opt = parser.parse_args()

_sep = 64 * "-"

# get path to the experiment directory
path = os.path.join(opt.root, opt.exp)
experiments = [e for e in os.listdir(path) if '.' != e[0]]

# prepare output diorectory
output_path = os.path.join(opt.output, opt.exp)
if not os.path.exists(output_path):
    os.makedirs(output_path)


# utilities
def _readable(key):
    return key.replace("loss/", "")


def _print_df(df):
    if opt.latex:
        print(df.to_latex(float_format=f"%{opt.float_format}"))
    else:
        print(df)


# define keys to read from the logs
valid_keys = [
    "elbo",
    "kl",
    "N_eff",
]
train_keys = [
    "control_variate_mse",
    "log_grad_var",
]
train_keys = ["loss/" + k for k in train_keys]
valid_keys = ["loss/" + k for k in valid_keys]

# read data
print("# reading experiments from path: ", path)
data = []
logs = []
for e in experiments:
    print(" - exp =", e)

    exp_path = os.path.join(path, e)

    files = os.listdir(exp_path)

    # TODO: use success flag to read or pass data
    try:
        # TODO: use success flag to read or pass data
        # with open(os.path.join(exp_path, 'success.txt'), 'r') as fp:
        #     success_ = fp.read()
        #
        # if "Success." in success_:
        if True:

            # reading configuration files with run parameter
            with open(os.path.join(exp_path, 'config.json'), 'r') as fp:
                args = DotMap(json.load(fp))

            # read training logs
            _dir = os.path.join(exp_path, 'train')
            _tf_log = [os.path.join(_dir, o) for o in os.listdir(_dir) if 'events.out.tfevents' in o][0]
            with open(_tf_log, 'rb') as f:
                reader = EventsFileReader(f)

                for item in reader:
                    step = item.step
                    for v in item.summary.value:
                        if v.tag in train_keys:
                            logs += [{'id': e, 'step': step, '_key': _readable(v.tag), '_value': float(v.simple_value)}]

            # read valid logs
            _dir = os.path.join(exp_path, 'valid')
            _tf_log = [os.path.join(_dir, o) for o in os.listdir(_dir) if 'events.out.tfevents' in o][0]
            best_metric = -1e20
            with open(_tf_log, 'rb') as f:
                reader = EventsFileReader(f)

                for item in reader:
                    step = item.step
                    for v in item.summary.value:
                        if v.tag in valid_keys:
                            logs += [{'id': e, 'step': step, '_key': _readable(v.tag), '_value': float(v.simple_value)}]
                        if v.tag == opt.metric:
                            best_metric = v.simple_value

            results = {'id': e, _readable(opt.metric): best_metric}

            # initializing experiment data as the argparse parameters
            d = dict(args)
            # append results to data
            d.update(results)

            # append data and logs
            data += [d]

    except Exception as ex:
        print("## FAILED. Exception:")
        print(_sep)
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        print(_sep)
        print("\nException: ", ex, "\n")

# compile data into a dataframe
df = pd.DataFrame(data)


print(df)

# drop columns that contain the same attributes (except for `seed`)
nunique = df.apply(pd.Series.nunique)
global_attributes = list(nunique[nunique == 1].index)
if 'seed' in global_attributes:
    global_attributes.remove('seed')
if _readable(opt.metric) in global_attributes:
    global_attributes.remove(_readable(opt.metric))
df = df.drop(global_attributes, axis=1)

# compile log data and merge with the attributes
logs = pd.DataFrame(logs)
_keys_to_merge = [k for k in df.keys() if k != _readable(opt.metric)]
logs = logs.merge(df[_keys_to_merge], left_on="id", right_on="id")

# drop id (exp name)
df = df.drop('id', 1)
logs = logs.drop('id', 1)

"""
print all results
"""

# sort by values
df = df.sort_values(_readable(opt.metric), ascending=False)

# print all results
print("\n" + _sep)
for g in global_attributes:
    print(f"{g} : {args[g]}")
print(_sep)
print(os.path.abspath(path))
print(_sep)
print("all data: varying parameters:", [k for k in df.keys() if k not in _readable(opt.metric)])
print(_sep)
if "dataset" in df.keys():
    for dset in set(df["dataset"].values):
        _print_df(df[df["dataset"] == dset])
        print(_sep)
else:
    _print_df(df)
    print()

"""
pivot table
"""

def aggfunc(serie):
    mean = np.mean(serie)
    std = np.std(serie)
    return f"{mean:{opt.float_format}} ± {std:{opt.float_format}} (n={len(serie)})"


_keys = [k for k in df.keys() if k != _readable(opt.metric) and k != "seed"]
pivot = df.pivot_table(index=_keys, values=_readable(opt.metric), aggfunc=aggfunc)
# sort pivot according to the mean value
mean_pivot = df.pivot_table(index=_keys, values=_readable(opt.metric), aggfunc=np.mean)
mean_pivot = mean_pivot.sort_values(_readable(opt.metric), ascending=False)
pivot = pivot.reindex(mean_pivot.index)

print("\n" + _sep)
print("Pivot table:\n" + _sep)
_print_df(pivot)
print(_sep)



"""
plot curves with uncertainty intervals
"""

# todo: allow selecting two keys for the anaylsis (e.g. estimator:nrow vs kdim:ncol)

_last_indexes = ['seed', '_key', 'step']
_index = [k for k in logs.keys() if k != '_value' and k not in _last_indexes]
_index += _last_indexes
_keys = logs['_key'].unique()

# get max step
M = logs['step'].max()
bins = [int(s) for s in range(0, M, M//opt.nsamples)]

n_full = len(logs['step'].unique())
ratio = n_full // opt.nsamples

# reshape data with index [..., seed, _key, step] and sort by steps
# logs = logs.pivot_table(index=_index, values='_value', aggfunc=np.mean)
# for idx in _index[::-1]:
#     logs.sort_index(level=idx, sort_remaining=False, inplace=True)

print(f"Downsampling data.. (n={opt.nsamples})")
logs.reset_index(level=-1, inplace=True)
_index.remove("step")
logs = logs.groupby(_index + [ pd.cut(logs.step, bins) ]).mean()
logs.index.rename(level=[-1], names=['step_bucket'], inplace=True)
logs.reset_index(inplace=True)

# drop nan
logs.dropna(inplace=True)

print("Generating plots")
N = len(_keys)
ncols = 2
nrows = N // ncols

if N > ncols * nrows:
    nrows += 1


hue_order = list(logs["estimator"].unique())
step_min = np.percentile(logs['step'].values.tolist(), 5)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
for i, k in tqdm(list(enumerate(_keys))):
    u = i // ncols
    v = i % ncols
    ax = axes[u, v]

    sns.lineplot(x="step", y="_value",
                 hue="estimator",
                 hue_order=hue_order,
                 data=logs[logs['_key']==k], ax=ax)

    ax.set_ylabel(k)
    # y lims
    ys = logs[(logs['_key'] == k) & (logs['step']>step_min)]['_value'].values.tolist()
    a, b = np.percentile(ys, [25, 75])
    M = b - a
    k = 1.5
    ax.set_ylim([a - k * M, b + k * M])
    ax.get_legend().remove()

# draw legend in the last plot
patches = [mpatches.Patch(color=sns.color_palette()[i], label=key) for i, key in enumerate(hue_order)]
axes[nrows-1, ncols-1].legend(handles=patches)

plt.tight_layout()
plt.savefig(os.path.join(output_path, f"curves.png"))
plt.close()