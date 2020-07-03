import argparse
import logging
import os
import shutil
import sys
import warnings
from multiprocessing import Pool

import GPUtil
from booster.utils import logging_sep
from tqdm import tqdm

from ovis.utils.dbutils import FileLockedTinyDB
from ovis.utils.dbutils import requeue_experiments
from ovis.utils.manager import snapshot_dir, read_experiment_json_file, get_abs_paths, retrieve_exp_and_run
from ovis.utils.utils import Header


def run_manager():
    """
    Run a set of experiments defined as a json file in `experiments/` using mutliprocessing.
    This script is a quick & dirty implementation of a queue system using `filelock` and `tinydb`.
    The manager creates a snapshot of the library to ensure consistency between runs. You may update the snapshot
    manually using `--update_lib` or update the experiment file using `--update_exp`.
    In that case starting a new manager using `--resume` will append potential new experiments to the database.
    Use `--rf` to delete an existing experiment. Example:

    ```bash
    python manager.py --exp gaussian-mixture-model --max_gpus 4 --processes 2
    ```

    Troubleshooting: If the `FilelockDB` exits without properly deleting the `.lock` file, you may need to delete
    it manually using the `--purge_lock`. This must be used carefully as it may break ongoing database transactions.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default='run.py',
                        help='script name')
    parser.add_argument('--root', default='runs/',
                        help='experiment directory')
    parser.add_argument('--data_root', default='data/',
                        help='data directory')
    parser.add_argument('--exp', default='gaussian-mixture-model', type=str,
                        help='experiment id pointing to a .json file in experiments/')
    parser.add_argument('--max_gpus', default=8, type=int,
                        help='maximum number of GPUs')
    parser.add_argument('--max_load', default=0.5, type=float,
                        help='only use GPUs with load < `max_load`')
    parser.add_argument('--max_memory', default=0.01, type=float,
                        help='only use GPUs with memory < `max_memory`')
    parser.add_argument('--processes', default=1, type=int,
                        help='number of processes per GPU')
    parser.add_argument('--rf', action='store_true', help='delete the previous experiment')
    parser.add_argument('--resume', action='store_true',
                        help='resume an experiment')
    parser.add_argument('--update_exp', action='store_true',
                        help='update snapshot experiment file')
    parser.add_argument('--update_lib', action='store_true',
                        help='update the entire snapshot')
    parser.add_argument('--max_jobs', default=-1, type=int,
                        help='maximum jobs per thread (stop after `max_jobs` jobs)')
    parser.add_argument('--requeue_level', default=1, type=int,
                        help='[db] Requeue level {0: nothing, 1: keyboard_interrupt, 2: failed, 3: not completed}')
    parser.add_argument('--purge_lock', action='store_true',
                        help='purge the `.lock` file [use only if the `.lock` file has not been properly removed].')
    opt = parser.parse_args()

    # get the list of devices
    device_ids = GPUtil.getAvailable(order='memory',
                                     limit=opt.max_gpus,
                                     maxLoad=opt.max_load,
                                     maxMemory=opt.max_memory,
                                     includeNan=False,
                                     excludeID=[],
                                     excludeUUID=[])
    if len(device_ids):
        device_ids = [f"cuda:{d}" for d in device_ids]
    else:
        device_ids = ['cpu']

    # total number of processes
    processes = opt.processes * len(device_ids)

    # get absolute path to logging directories
    exps_root, exp_root, exp_data_root = get_abs_paths(opt.root, opt.exp, opt.data_root)

    if os.path.exists(exp_root):
        warnings.warn(f"logging directory `{exp_root}` already exists.")

        if not opt.resume:
            if opt.rf:
                warnings.warn(f"Deleting existing logging directory `{exp_root}`.")
                shutil.rmtree(exp_root)
            else:
                sys.exit()

    # copy library to the `snapshot` directory
    if not opt.resume:
        shutil.copytree('./', snapshot_dir(exp_root),
                        ignore=shutil.ignore_patterns('.*', '*.git', 'runs', 'reports', 'data', '__pycache__'))

    if opt.update_lib:
        # move original lib
        shutil.move(snapshot_dir(exp_root), f"{snapshot_dir(exp_root)}-saved")

        # copy lib
        shutil.copytree('./', snapshot_dir(exp_root),
                        ignore=shutil.ignore_patterns('.*', '*.git', 'runs', 'reports', 'data', '__pycache__', '*.psd'))

    # udpate experiment file
    if opt.update_exp:
        _exp_file = f'experiments/{opt.exp}.json'
        shutil.copyfile(_exp_file, os.path.join(snapshot_dir(exp_root), _exp_file))

    # move path to the snapshot directory to ensure consistency between runs (lib will be loaded from `./lib_snapshot/`)
    os.chdir(snapshot_dir(exp_root))

    # logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger('exp-manager')

    # log config
    print(logging_sep("="))
    logger.info(
        f"Available devices: {device_ids}")
    logger.info(
        f"Experiment id = {opt.exp}, running {opt.processes} processes/device, logdir = {exp_root}")
    print(logging_sep())

    # read the experiment file
    experiment_args = read_experiment_json_file(opt.exp)

    # replace the default `script` argument if specified
    if "script" in experiment_args.keys():
        opt.script = experiment_args.pop("script")

    # define the arguments for each run
    args = experiment_args['args']  # retrieve the list of arguments
    args = [experiment_args['global'] + " " + a for a in args]  # append the `global` arguments
    args = [f"--exp {opt.exp} --root {exps_root} --data_root {exp_data_root} {a}" for a in
            args]  # append specific parameters
    if "parameters" in experiment_args.keys():
        for _arg, values in experiment_args["parameters"].items():
            _args = []
            for v in values:
                if isinstance(v, bool):
                    if v:
                        _args += [f"--{_arg} " + a for a in args]
                    else:
                        _args += args
                else:
                    _args += [f"--{_arg} {v} " + a for a in args]
            args = _args

    # remove the lock file manually
    if opt.purge_lock:
        FileLockedTinyDB(exp_root).purge()

    # write all experiments to `tinydb` database guarded by a `filelock`
    with FileLockedTinyDB(exp_root) as db:
        query = db.query()
        # add missing exps (when using `--resume` + `--update_exp`)
        n_added = 0
        for i, a in enumerate(args):
            if not db.contains(query.arg == a):
                db.insert({'arg': a, 'queued': True, "job_id": "none"})
                n_added += 1

    # potentially check the database status requeue fail experiment
    if opt.script == 'run.py':  # only handled for `run.py` because matching exps relies on `get_run_parser`
        with Header(f"Status"):
            requeue_experiments(exp_root, opt.requeue_level)

    # count queued exps and total exps
    with FileLockedTinyDB(exp_root) as db:
        n_queued_exps = db.count(query.queued == True)
        n_exps = len(db)

    # remaining queued experiments
    logger.info(f"Queued experiments : {n_queued_exps} / {n_exps}. Added exps. {n_added}")

    # run processes in parallel (spawning `processes` processes)
    pool = Pool(processes=processes)
    job_args = [{"opt": opt, "exp_root": exp_root, "devices": device_ids} for _ in range(n_queued_exps)]

    if opt.max_jobs > 0:
        job_args = job_args[:opt.max_jobs * processes]

    logger.info(f"Max. number of jobs = {len(job_args)}, processes = {processes} [{len(device_ids)} devices]")
    for _ in tqdm(pool.imap_unordered(retrieve_exp_and_run, job_args, chunksize=1), total=n_queued_exps,
                  desc="Job Manager"):
        pass


if __name__ == '__main__':
    run_manager()
