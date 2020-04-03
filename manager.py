import argparse
import logging
import multiprocessing
import os
import shutil
import socket
import sys
import traceback
import warnings
from multiprocessing import Pool

import GPUtil
from filelock import filelock  # pip installl git+https://github.com/dmfrey/FileLock.git
from tqdm import tqdm

from lib.manager import open_db, snapshot_dir, read_experiment, get_abs_paths, get_filelock


def fn(job_args):
    opt, abs_logdir, devices = [job_args[k] for k in ["opt", "exp_root", "devices"]]
    _max_attempts = 10
    _hostname = socket.gethostname()
    _pid = os.getpid()
    _jobid = f"{_hostname}-{_pid}"

    # retrieve the next queued experiment from the database
    # lock db file to avoid concurrency problems
    with filelock.FileLock(get_filelock(abs_logdir)):
        db, query = open_db(abs_logdir)
        item = db.get(query.queued == True)
        if item is not None:
            item['queued'] = False
            db.write_back([item])
        del db

    if item is None:
        print(f"Manager: no job left.")
        return None
    else:
        args = item['arg']

        process_id = eval(multiprocessing.current_process().name.split('-')[-1]) - 1
        device = devices[process_id % len(devices)]
        print(
            f"initializing process with PID = {os.getpid()}, process id: {process_id}, allocated device: {device}, args= {args}")

        if 'cuda' in device:
            device_id = device.split(':')[-1]
            command = f"CUDA_VISIBLE_DEVICES={device_id} python {opt.script} {args}"
        else:
            command = f"python {opt.script} {args}"
        try:
            os.system(command)
        except Exception as ex:
            print(f"Command `{command}` failed.")
            print("-------------------------------------------------")
            traceback.print_exception(type(ex), ex, ex.__traceback__)

        print(f"{process_id} DONE. \nargs = {args}\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default='run.py', help='script name')
    parser.add_argument('--root', default='runs/', help='experiment directory')
    parser.add_argument('--data_root', default='data/', help='data directory')
    parser.add_argument('--exp', default='estimators-0.1', type=str, help='experiment id')
    parser.add_argument('--max_gpus', default=8, type=int, help='maximum number of gpus')
    parser.add_argument('--max_load', default=0.5, type=float, help='maximum GPU load')
    parser.add_argument('--max_memory', default=0.01, type=float, help='maximum GPU memory')
    parser.add_argument('--processes', default=1, type=int, help='number of processes per GPU')
    parser.add_argument('--rf', action='store_true', help='force delete previous experiment')
    parser.add_argument('--append', action='store_true', help='force append new experiment')
    opt = parser.parse_args()

    # get available devices
    deviceIDs = GPUtil.getAvailable(order='load', limit=opt.max_gpus, maxLoad=opt.max_load, maxMemory=opt.max_memory,
                                    includeNan=False, excludeID=[], excludeUUID=[])
    if len(deviceIDs):
        deviceIDs = [f"cuda:{d}" for d in deviceIDs]
    else:
        deviceIDs = ['cpu']

    # total number of processes
    processes = opt.processes * len(deviceIDs)

    # get absolute path to logging directories
    exps_root, exp_root, exp_data_root = get_abs_paths(opt.root, opt.exp, opt.data_root)

    if os.path.exists(exp_root):
        warnings.warn(f"logging directory `{exp_root}` already exists.")

        if not opt.append:
            if opt.rf:
                warnings.warn(f"Deleting existing logging directory `{exp_root}`.")
                shutil.rmtree(exp_root)
            else:
                sys.exit()

    # copy library to the `snapshot` directory
    if not opt.append:
        shutil.copytree('./', snapshot_dir(exp_root),
                        ignore=shutil.ignore_patterns('.*', '*.git', 'runs', 'reports', 'data'))

    # move path to the snapshot directory to ensure consistency between runs (lib will be loaded from `./lib_snapshot/`)
    os.chdir(snapshot_dir(exp_root))

    # init db
    db, query = open_db(exp_root)

    # logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger('exp-manager')

    # log config
    logger.info(
        f"Available devices: {deviceIDs}")
    logger.info(
        f"Experiment id = {opt.exp}, running {opt.processes} processes/device, logdir = {exp_root}")

    experiment_args = read_experiment(opt.exp)

    # define the args for each run
    args = experiment_args['args']

    # dropout and others
    args = [experiment_args['global'] + " " + a for a in args]

    # append logdir to args
    args = [f"--exp {opt.exp} --root {exps_root} --data_root {exp_data_root} {a}" for a in args]

    # parameters lists
    if "parameters" in experiment_args.keys():
        for _arg, values in experiment_args["parameters"].items():
            _args = []
            for v in values:
                _args += [f"--{_arg} {v} " + a for a in args]
            args = _args

    # for i, a in enumerate(args):
    #     logger.info(f"# EXPERIMENT #{i}:  {a}")

    # todo: write all experiments to a database
    # for i, a in enumerate(args):
    #     if not db.contains(query.arg == a):
    #         db.insert({'arg': a, 'queued': True, "job_id": "none"})

    n_records = len(db.search(query.queued))

    logger.info(
        f"Database: queued experiments : {n_records}, total experiments: {len(db)}")

    # run processes in parallel
    pool = Pool(processes=processes)
    job_args = [{"opt": opt, "exp_root": exp_root, "devices": deviceIDs} for _ in range(n_records)]
    for _ in tqdm(pool.imap_unordered(fn, job_args, chunksize=1), total=n_records, desc="Job Manager"):
        pass
