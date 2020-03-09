import argparse
import logging
import multiprocessing
import os
import shutil
import sys
import json
import warnings
from multiprocessing import Pool

import GPUtil

def fn(allargs):
    args, devices = allargs

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
    except:
        print(f"Command `{command}` failed.")

    print(f"{process_id} DONE.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default='run.py', help='script name')
    parser.add_argument('--root', default='runs/', help='experiment directory')
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

    # log directory
    logdir = os.path.join(opt.root, opt.exp)
    if os.path.exists(logdir):
        warnings.warn(f"logging directory `{logdir}` already exists.")

        if not opt.append:
            if opt.rf:
                warnings.warn(f"Deleting existing logging directory `{logdir}`.")
                shutil.rmtree(logdir)
            else:
                sys.exit()

    # logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    logger = logging.getLogger('exp-manager')

    # log config
    logger.info(
        f"Available devices: {deviceIDs}")
    logger.info(
        f"Experiment id = {opt.exp}, running {opt.processes} processes/device, logdir = {logdir}")

    with open(f'exps/{opt.exp}.json') as json_file:
        data = json.load(json_file)

    # define the args for each run
    args = data['args']

    # dropout and others
    args = [data['global'] + " " + a for a in args]

    # append logdir to args
    args = [f"--exp {opt.exp} --root {opt.root} " + a for a in args]

    # parameters lists
    if "parameters" in data.keys():
        for _arg, values in data["parameters"].items():
            _args = []
            for v in values:
                _args += [f"--{_arg} {v} " + a for a in args]
            args = _args

    for i, a in enumerate(args):
        logger.info(f"# EXPERIMENT #{i}:  {a}")

    # run processes in parallel
    pool = Pool(processes=processes)
    devices_args = [deviceIDs for _ in args]
    pool.map(fn, zip(args, devices_args), chunksize=1)
