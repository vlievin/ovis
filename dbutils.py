from ovis.utils.dbutils import *
from ovis.utils.manager import get_abs_paths
from ovis.utils.utils import Header

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
                    help='check experiments status')
parser.add_argument('--error_messages', action='store_true',
                    help='display error messages')
parser.add_argument('--requeue', action='store_true',
                    help='requeue experiment according to ´--requeue_level´')
parser.add_argument('--requeue_level', default=1, type=int,
                    help='Requeue level {0: only check, '
                         '1: aborted_by_user (sigterm, keyboard_interrupt), not_found (no exp file was found)'
                         '2: failed, '
                         '100: requeue `running` (without `success` file), '
                         '10000: all runs, including successful ones}')
opt = parser.parse_args()

# get absolute path to logging directories
exps_root, logdir, _ = get_abs_paths(opt.root, opt.exp, None)

# Display the number of records and db path
with FileLockedTinyDB(logdir) as db:
    with Header(message=None):
        query = Query()
        queued = db.count(query.queued == True)
        print(f"[Experiments] Queued = {queued}, Total = {len(db)}, path = {logdir}")

if opt.show_all:
    with Header("All records"):
        show_all_experiments(logdir)

if opt.find:
    with Header(f"Query = `{opt.find}`"):
        query = Query()
        find_experiments(logdir, eval(opt.find))

if len(opt.update):
    with Header(f"Updating records with pattern `{opt.update}`"):
        update_experiments(logdir, opt.update)

if len(opt.delete) > 0:
    with Header(f"Deleting records matching pattern `{opt.delete}`"):
        delete_experiments(logdir, opt.delete)

if opt.check:
    with Header(f"Status"):
        requeue_experiments(logdir, level=0)

if opt.error_messages:
    with Header(f"Error Messages"):
        requeue_experiments(logdir, level=0, display_mode='messages')

if opt.requeue:
    with Header(f"Requeuing"):
        requeue_experiments(logdir, level=opt.requeue_level)
