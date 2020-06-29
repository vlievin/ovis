import json
import os

from tinydb import TinyDB, Query  # pip install tinydb


def open_db(logdir):
    _file = os.path.join(logdir, '.db.json')
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
