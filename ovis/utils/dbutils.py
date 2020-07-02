import json
import os
from collections import defaultdict
from functools import partial

from booster.utils import logging_sep
from tinydb import TinyDB, Query

from ovis.training.arguments import get_run_parser
from ovis.training.utils import get_hash_from_opt
from ovis.utils.filelock import FileLock
from ovis.utils.utils import Success


class FileLockedTinyDB(FileLock, TinyDB):
    """
    A TinyDB json database guarded by a `filelock`. Usage:
    ```python
    logdir = './'
    # write data
    with FileLockedTinyDB(logdir) as db:
        query = Query()
        for a in ["--dataset aa", "--dataset bb"]:
            if not db.contains(query.arg == a):
                    db.insert({'arg': a, 'queued': True})

    # retrieve data
    with FileLockedTinyDB(logdir) as db:
        query = db.query()
        print(db.search(query.queued))
    ```
    """
    default_indent = 4
    default_timeout = 10
    default_delay = 0.5
    default_file_name = '.db.json'

    def __init__(self, logdir, *args, **kwargs):
        indent = kwargs.pop('indent', self.default_indent)
        timeout = kwargs.pop('timeout', self.default_timeout)
        delay = kwargs.pop('delay', self.default_delay)
        path = os.path.join(logdir, self.default_file_name)
        TinyDB.__init__(self, path, *args, indent=indent, **kwargs)
        FileLock.__init__(self, path, timeout=timeout, delay=delay)

    @staticmethod
    def query():
        return Query()

    def rf_lock(self):
        os.remove(self.lockfile)

    def __enter__(self):
        FileLock.__enter__(self)
        return TinyDB.__enter__(self)

    def __exit__(self, *args):
        TinyDB.__exit__(self, *args)
        FileLock.__exit__(self, *args)


def print_record(index, record):
    print(f"{index + 1}# : queued = {record['queued']}, arg = \n     {record['arg']}")


def show_all_records(logdir):
    """print all records in the database"""
    with FileLockedTinyDB(logdir) as db:
        for i, record in enumerate(db.all()):
            print_record(i, record)


def find_records(logdir, query_statement):
    """print all records matching the query"""
    with FileLockedTinyDB(logdir) as db:
        search_results = db.search(query_statement)
        print(f"Found {len(search_results)} matching records\n{logging_sep()}")
        for i, record in enumerate(search_results):
            print_record(i, record)


def update_records(logdir, rules):
    """update the records in the database based on rules where rules follow the format `key:old_value,key:new_value`"""
    with FileLockedTinyDB(logdir) as db:
        query = db.query()
        for rule in rules.split(','):
            key, arg, new_arg = rule.split(':')
            pattern = f"--{key} {arg}"
            new_pattern = f"--{key} {new_arg}"
            matching_exps = db.search(query.arg.test(lambda x: pattern in x))
            print(logging_sep())
            print(f"Updating {len(matching_exps)} records with key = {key} : {arg} -> {new_arg}")
            for exp in matching_exps:
                exp['arg'] = exp['arg'].replace(pattern, new_pattern)
            db.write_back(matching_exps)


def delete_records(logdir, pattern):
    """delete experiments that match some pattern"""
    with FileLockedTinyDB(logdir) as db:
        query = db.query()
        results = db.search(query.arg.test(lambda x: pattern in x))
        print(logging_sep("-"))
        print(f"Deleting {len(results)} / {len(db)} records")
        db.remove(query.arg.test(lambda x: pattern in x))


def get_hash_from_record(parser, record):
    db_opt = parser.parse_args(record['arg'].split(' '))
    return get_hash_from_opt(db_opt)


def requeue_records(logdir, level=1):
    """
    check queued==False records, find there corresponding experiment folder and
    requeue (i.e. set record.queue==True) based on the `level`value. Levels:
    * 0: do not requeue
    * 1: requeue `keyboard_interrupt`
    * 2: requeue `failed``
    * 100: requeue `running`
    * 10000: all including `success`
    """

    def requeue(queue, record, exp):
        """set record to `queued` flag to True, append record to the ´to_be_requeued´ queue and delete the ´success´ file"""
        record['queued'] = True
        queue += [record]
        os.remove(os.path.join(logdir, exp, Success.file))

    with FileLockedTinyDB(logdir) as db:
        query = db.query()
        parser = get_run_parser()
        get_hash = partial(get_hash_from_record, parser)
        status = defaultdict(lambda: 0)
        requed_status = defaultdict(lambda: 0)

        # count queued records
        status['queued'] = db.count(query.queued == True)
        # retrieve all non-queued records from the db and get the run_id hash
        db_hashes = {get_hash(record): record for record in db.search(query.queued == False)}
        db_hashes_set = set(db_hashes.keys())
        # iterate through experiments files and store the experiments that have been interrupted
        to_be_requeued = []
        for exp in [e for e in os.listdir(logdir) if e[0] != '.']:
            with open(os.path.join(logdir, exp, 'config.json'), 'r') as f:
                exp_config = json.load(f)
                exp_hash = exp_config['hash']
                if exp_hash in db_hashes_set:
                    if Success.file in os.listdir(os.path.join(logdir, exp)):
                        message = open(os.path.join(logdir, exp, Success.file), 'r').read()

                        if Success.keyboard_interrupt == message:
                            status['keyboard_interrupt'] += 1
                            if level >= 1:
                                requed_status['keyboard_interrupt'] += 1
                                requeue(to_be_requeued, db_hashes[exp_hash], exp)

                        elif Success.success == message:
                            if level >= 10000:
                                requed_status['success'] += 1
                                requeue(to_be_requeued, db_hashes[exp_hash], exp)
                            status['success'] += 1

                        elif Success.failure_base in message:
                            status['failed'] += 1
                            if level >= 2:
                                requed_status['failed'] += 1
                                requeue(to_be_requeued, db_hashes[exp_hash], exp)
                        else:
                            raise ValueError(f"Couldn't handle the success message `{message}`")
                    else:
                        status['running'] += 1
                        if level >= 100:
                            requed_status['running'] += 1
                            requeue(to_be_requeued, db_hashes[exp_hash], exp)

        # requeue the stored experiments
        db.write_back(to_be_requeued)

        # print status
        status['queued'] += sum([v for k, v in requed_status.items() if k != 'queued'])
        for k, v in status.items():
            print(f"  [{k}] {v - requed_status[k]} Records (Requeud: {requed_status[k]})")
