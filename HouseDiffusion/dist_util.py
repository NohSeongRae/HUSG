import io
import os
import socket
import blobfile as bf
from mpi4py import MPI
import torch
import torch.distributed as dist

GPUS_PER_NODE =4

def setup_dist():
    pass

def dev():
    pass

def load_state_dict(path, **kwargs):
    pass

def sync_params(params):
    pass

def _find_free_port():
    pass