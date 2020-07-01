import json
import multiprocessing
import os
import socket
import time
import traceback

from ovis.utils.dbutils import FileLockedTinyDB


def snapshot_dir(logdir):
    return os.path.join(logdir, '.lib_snapshot')


def get_abs_paths(root, exp, data_root):
    exps_root = os.path.abspath(root)  # e.g. runs/
    exp_root = os.path.abspath(os.path.join(exps_root, exp))  # e.g. runs/experiment/
    exp_data_root = os.path.abspath(data_root) if data_root is not None else None  # e.g. data/
    return exps_root, exp_root, exp_data_root


def read_experiment_json_file(exp):
    with open(f'experiments/{exp}.json') as json_file:
        experiment_args = json.load(json_file)

    return experiment_args


def retrieve_exp_and_run(job_args):
    """
    Retrieve an experiment from the database, run, iterate.
    :param job_args: arguments for this subprocess
    """
    opt, abs_logdir, devices = [job_args[k] for k in ["opt", "exp_root", "devices"]]
    _max_attempts = 100
    _attempts = 0
    _wait_time = 2
    _success = False
    _hostname = socket.gethostname()
    _pid = os.getpid()
    _jobid = f"{_hostname}-{_pid}"

    while not _success and _attempts < _max_attempts:
        try:
            # retrieve the next queued experiment from the database and mark `queued==False`
            # lock db file to avoid concurrency problems
            with FileLockedTinyDB(abs_logdir) as db:
                query = db.query()
                item = db.get(query.queued == True)
                if item is not None:
                    item['queued'] = False
                    db.write_back([item])
                del db
                time.sleep(0.2)
            _success = True

            if item is None:
                print(f"@ manager.subprocess : no job left.")
                exit()
            else:
                args = item['arg']
                process_id = eval(multiprocessing.current_process().name.split('-')[-1]) - 1
                device = devices[process_id % len(devices)]
                print(f"@ manager.subprocess : initializing process with PID = {os.getpid()}, "
                      f"process id: {process_id}, allocated device: {device}, args= {args}")

                if 'cuda' in device:
                    device_id = device.split(':')[-1]
                    if os.name == 'nt':
                        command = f"set CUDA_VISIBLE_DEVICES={device_id} & python {opt.script} {args}"
                    else:
                        command = f"CUDA_VISIBLE_DEVICES={device_id} python {opt.script} {args}"
                else:
                    command = f"python {opt.script} {args}"
                try:
                    os.system(command)
                except:
                    print(f"Command `{command}` failed.")

                print(
                    f"@ manager.subprocess [id={process_id}, pid={_pid}] : experiment completed. Arguments = \n{args}\n\n")

        except KeyboardInterrupt:
            print(f"@ manager.subprocess : Keyboard Interrupt")

        except Exception as ex:
            _attempts += 1
            print(f"#@ manager.subprocess : failed. Attempt = {_attempts} / {_max_attempts}")
            print("--------------------------------------------------------------------------------")
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            print("--------------------------------------------------------------------------------")
            time.sleep(_wait_time)
