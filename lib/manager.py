import json
import os
from tinydb import TinyDB, Query  # pip install tinydb


def open_db(logdir):
    _file = os.path.join(logdir, '.db.json')

    # sometimes the file get corrupted with trailing null bytes (\0x00)
    # remove them as an ugly and quick workaround that works in some cases
    # sometimes it still fails inconsistently: this must be due to the shared
    # file system. Using /nocbackup may solve the issue.
    # with filelock.FileLock(get_filelock(logdir)):
    #     with open(_file, 'r') as fp:
    #         s = fp.read()
    #
    #     if s.find('\x00'):
    #         print("# manager.open_db: removing null bytes.")
    #         s = s.replace('\x00', '')
    #         with open(_file, 'w') as fp:
    #             fp.write(s)

    db = TinyDB(_file, indent=4)
    query = Query()
    return db.table('experiments'), query


def get_filelock(abs_logdir):
    return os.path.join(abs_logdir, ".db.json")

def snapshot_dir(logdir):
    return os.path.join(logdir, '.lib_snapshot')

def get_abs_paths(root, exp, data_root):
    exps_root = os.path.abspath(root)  # e.g. runs/
    exp_root = os.path.abspath(os.path.join(exps_root, exp))  # e.g. runs/experiment/
    exp_data_root = os.path.abspath(data_root) if data_root is not None else None  # e.g. data/
    return exps_root, exp_root, exp_data_root


def read_experiment(exp):
    with open(f'exps/{exp}.json') as json_file:
        experiment_args = json.load(json_file)

    return experiment_args
