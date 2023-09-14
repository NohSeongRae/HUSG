import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50

class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        pass
    def writekvs(self, kvs):
        pass
    def _truncate(self, s):
        pass
    def writeseq(self, seq):
        pass
    def close(self):
        pass

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        pass
    def writekvs(self, kvs):
        pass
    def close(self):
        pass
class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        pass
    def writekvs(self, kvs):
        pass
    def close(self):
        pass

class TensorBoardOutputFormat(KVWriter):
    def __init__(self, dir):
        pass
    def writekvs(self, kvs):
        pass
    def close(self):
        pass

def make_output_format(format, ev_dir, log_suffix=""):
    pass

def logkv(key, val):
    pass

def logkv_mean(key, val):
    pass

def logkvs(d):
    pass

def dumpkvs():
    pass

def getkvs():
    pass

def log(*args, level=INFO):
    pass

def debug(*args):
    pass

def info(*args):
    pass

def warn(*args):
    pass
def error(*args):
    pass

def set_leve(level):
    pass

def set_comm(comm):
    pass

def get_dir():
    pass

record_tabular=logkv
dump_tabular=dumpkvs()

@contextmanager
def profile_kv(scopename):
    pass

def profile(n):
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            pass
        pass
    pass


def get_current():
    pass

class Logger(object):
    DEFAULT=None
    CURRENT=None
    def __init__(self):
        pass
    def logkv(self, key, val):
        pass
    def logkv_mean(self, key, val):
        pass
    def dumpkvs(self):
        pass
    def log(self, *args, level=INFO):
        pass
    def set_leve(self, level):
        pass
    def set_comm(self, comm):
        pass
    def get_dir(self):
        pass
    def close(self):
        pass
    def _do_log(self, args):
        pass

def get_rank_without_mpi_import():
    pass

def mpi_weighted_mean(comm, local_name2valcount):
    pass

def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    pass

def _configure_defualt_logger():
    pass

def reset():
    pass

@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    pass





