import argparse

from ovis.utils.dbutils import *
from ovis.utils.manager import get_abs_paths

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/',
                    help='experiment directory')
parser.add_argument('--exp', default='gaussian-mixture-model',
                    help='experiment id')
parser.add_argument('--update', default='',
                    help='comma separated list of args :key:value:new_value, e.g. `estimator:ovis-gamma1:reinforce`')
parser.add_argument('--delete', default='',
                    help='pattern to be match in exp')
parser.add_argument('--show_all', action='store_true',
                    help='show all records')
parser.add_argument('--find', default='',
                    help='show results from query. '
                         'Examples: --find "query.queued == True", --find "query.arg.search(\'(?=.*ovis)\')"')
parser.add_argument('--check', action='store_true',
                    help='check potential failed experiments')
parser.add_argument('--requeue', action='store_true',
                    help='requeue experiment according to ´--requeue_level´')
parser.add_argument('--requeue_level', default=1, type=int,
                    help='Requeue level {0: nothing, 1: keyboard_interrupt, 2: failed, 3: not completed}')
opt = parser.parse_args()

_sep = os.get_terminal_size().columns * "-"

# get absolute path to logging directories
exps_root, logdir, _ = get_abs_paths(opt.root, opt.exp, None)

# Display the number of records and db path
with FileLockedTinyDB(logdir) as db:
    print(logging_sep("="))
    query = Query()
    queued = db.count(query.queue == True)
    print(f"Queued Records = {queued}, Total Records = {len(db)}, path = {logdir}")
    print(logging_sep("=") + "\n")

if opt.show_all:
    with Header("All records"):
        show_all_records(logdir)

if opt.find:
    with Header(f"Query = `{opt.find}`"):
        query = Query()
        find_records(logdir, eval(opt.find))

if len(opt.update):
    with Header(f"Updating records with pattern `{opt.update}`"):
        update_records(logdir, opt.update)

if len(opt.delete) > 0:
    with Header(f"Deleting records matching pattern `{opt.delete}`"):
        delete_records(logdir, opt.delete)

if opt.check:
    with Header(f"Status"):
        requeue_records(logdir, level=0)

if opt.requeue:
    with Header(f"Requeuing"):
        requeue_records(logdir, level=opt.requeue_level)
