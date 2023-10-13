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
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                    "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create string for printing
        key2str = {}
        for (key, val) in sorted(kvs.item()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.item(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readline()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import _compat_pickle

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step(
            self.step
        )
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    """

    :param format: format for logging
    :param ev_dir: log save dir
    :param log_suffix: suffix string added to log file
    :return: logfile
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


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


record_tabular = logkv
dump_tabular = dumpkvs()


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
    DEFAULT = None  # A logger with no output files.
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1)) for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1
        out = d.copy()
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # configuration
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def get_rank_without_mpi_import():
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def mpi_weighted_mean(comm, local_name2valcount):
    """
        Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
        Perform a weighted average over dicts that are each on a different node
        Input: local_name2valcount: dict mapping key -> (value, count)
        Return: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warm(
                            f"WARNING: tried to compute mean on non-float {name} = {val}"
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
            return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    """
    if comm is provided, average all numerical stats across that comm
    :param dir: directory for logging
    :param format_strs: formats for logs
    :param comm: a communicator for distributed logging
    :param log_suffix:a suffix to be added to the log filenames
    :return: None
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
        dir = osp.join(
            'ckpts',
            datetime.datetime.now().strftime("openai_%Y_%m_%d_%H_%M_%S_%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging it %s" % dir)


def _configure_defualt_logger():
    pass


def reset():
    pass


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    pass