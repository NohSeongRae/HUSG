import io
import os
import socket
import blobfile as bf
from mpi4py import MPI
import torch
import torch.distributed as dist

GPUS_PER_NODE =4
SETUP_RETRY_COUNT=3
def setup_dist():
    """
    Set up a distributed process group
    :return: None
    """
    if dist.is_initialized():
        return
    comm=MPI.COMM_WORLD
    backend="gloo" if not torch.cuda.is_available() else "nccl"

    if backend=="gloo":
        hostname="localhost"
    else:
        hostname=socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"]=comm.bcast(hostname, root=0)
    os.environ["RANK"]=str(comm.rank)
    os.environ["WORLD_SIZE"]=str(comm.size)

    port=comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"]=str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    pass

def load_state_dict(path, **kwargs):
    pass

def sync_params(params):
    pass

def _find_free_port():
    try:
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()