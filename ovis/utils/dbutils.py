import os

from tinydb import TinyDB, Query


def open_db(logdir):
    _file = os.path.join(logdir, '.db.json')
    db = TinyDB(_file, indent=4)
    query = Query()
    return db.table('experiments'), query


def get_filelock(abs_logdir):
    return os.path.join(abs_logdir, ".db.json")