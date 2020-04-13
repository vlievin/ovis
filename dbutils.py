import sys
import os
import json
import argparse
import shutil
import traceback
from filelock import filelock
from lib.manager import open_db, snapshot_dir, read_experiment, get_abs_paths, get_filelock

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/nobackup/valv/runs/copt/', help='experiment directory')
parser.add_argument('--exp', default='binary-images-0.3', type=str, help='experiment id')
parser.add_argument('--update', default='', help='comma separated list of args :key:value:new_value, eg `data:path:newpath,iw:10:100`')
parser.add_argument('--delete', default='', help='pattern to be match in exp')
parser.add_argument('--show', action='store_true', help='show all records')
parser.add_argument('--check', action='store_true', help='check potential failed experiments')
parser.add_argument('--requeue', action='store_true', help='requeue experiment without successful outcome in `success.txt`')
parser.add_argument('--hard_requeue', action='store_true', help='also recover experiments without the `success` file. Warning: will delete runs of currently running exps.')
opt = parser.parse_args()


print(f"## dbutils: root = {opt.root}")

_sep = os.get_terminal_size().columns * "-"

_success_flag = "Success"

# get absolute path to logging directories
exps_root, exp_root, _ = get_abs_paths(opt.root, opt.exp, None)

try:
    with filelock.FileLock(get_filelock(exp_root), timeout=10):
        db, query = open_db(exp_root)
except Exception as ex:
    print("--------------------------------------------------------------------------------")
    traceback.print_exception(type(ex), ex, ex.__traceback__)
    print("--------------------------------------------------------------------------------")
    print(f"Python = {sys.version} ")
    exit()

print(_sep)
print(f"Records = {len(db)}, path = {exp_root}")

if opt.show:
    """show all records from the database"""
    print(_sep)
    for i, record in enumerate(db.all()):
        print(f"#{i+1}: queued = {record['queued']}, arg = \n{record['arg']}")

# show number of queued experiments
queued_exps = db.search(query.queued==True)
print(_sep)
print(f"Queued exps = {len(queued_exps)} / {len(db)}")

if len(opt.update):
    """update experiments args in the database"""
    for rule in opt.update.split(','):
        key, arg, new_arg = rule.split(':')
        pattern = f"--{key} {arg}"
        new_pattern = f"--{key} {new_arg}"
        matching_exps = db.search(query.arg.test(lambda x: pattern in x))
        print(_sep)
        print(f"Updating {len(matching_exps)} records with key = {key} : {arg} -> {new_arg}")
        for exp in matching_exps:
            exp['arg'] = exp['arg'].replace(pattern, new_pattern)
        db.write_back(matching_exps)

# delete experiments
if len(opt.delete) > 0:
    """delete experiments that match some pattern"""
    print(f" deleting entries matchin pattern {opt.delete}")
    results = db.search(query.arg.test(lambda x: opt.delete in x))
    print(_sep)
    print(f"Deleting {len(results)} / {len(db)}")
    db.remove(query.arg.test(lambda x: opt.delete in x))


if opt.check or opt.requeue:
    """Find the jobs that have failed (they were aborted by user of the server crashed), delete the actual records and requeue them in the database"""

    success_flag = "Success"
    db, query = open_db(exp_root)

    # move path to snapshot directory
    os.chdir(snapshot_dir(exp_root))

    exp_args = read_experiment(opt.exp)

    # find the arguments to focus on
    focus = list(exp_args['parameters'].keys())

    for a in exp_args['args']:
        focus += [aa.replace("--", "") for aa in a.split(" ") if "--" in aa]

    # remove doublons
    focus = list(set(focus))
    print(_sep)
    print("# Requeuing by matching args on keys:", focus)
    print(_sep)

    i = 0
    n_requeued = 0
    for exp in os.listdir(exp_root):

        if exp[0] != '.':

            exp_path = os.path.join(exp_root, exp)
            success_file = 'success.txt'

            # check if experiment needs to be recovered
            recover_exp = False
            if success_file in os.listdir(exp_path):
                with open(os.path.join(exp_path, success_file), 'r')  as fp:
                    outcome = fp.read()

                recover_exp = not _success_flag in outcome

            else:
                if opt.hard_requeue:
                    recover_exp = True

            # recover experiment in database
            if recover_exp:
                with open(os.path.join(exp_path, 'config.json'), 'r') as fp:
                    exp_opt = json.loads(fp.read())

                # build dict of args from the experiment config using the focus keys
                exp_args = {}
                for key in focus:
                    value = exp_opt[key]
                    if isinstance(value, bool):
                        if value:
                            exp_args[key] = 'true'

                    else:
                        exp_args[key] = str(value)


                def check(dbarg):
                    # parse the attribute `arg` in the database: e.g. arg = --dataset binmnist --N 16 --silent
                    def _extend(a):
                        kv = [b for b in a.split(" ") if b != ""]
                        if len(kv) == 1:
                            kv = kv[0], 'true'
                        return kv

                    dbsargs = [_extend(a) for a in dbarg.split("--")]

                    # transform into a dict with the focus keys
                    dbsargs = {u[0]: u[1] for u in dbsargs if (len(u) == 2) and u[0] in focus}

                    # check that `dbsargs` match with `exp_args`
                    return all(dbsargs[k] == v for k, v in exp_args.items() if k in dbsargs)


                results = db.search(query.arg.test(check))
                if len(results) != 1:
                    print(results)
                assert len(results) == 1

                # re-queue experiments
                for r in results:
                    r['queued'] = True
                    n_requeued += 1

                if opt.requeue:

                    db.write_back(results)
                    # delete experiment run
                    print("Deleting:", exp_path)
                    shutil.rmtree(exp_path)

    print(_sep)
    if opt.requeue:
        print(f"Requeued {n_requeued} experiments.")
    else:
        print(f"{n_requeued} experiments requeuable experiments.")
    queued_exps = db.search(query.queued == True)
    print(_sep)
    print(f"Queued exps = {len(queued_exps)} / {len(db)}")

print(_sep)