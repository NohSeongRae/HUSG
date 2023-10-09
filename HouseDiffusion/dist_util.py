import io
import os
import socket
import blobfile as bf
from mpi4py import MPI
import torch
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE =4
SETUP_RETRY_COUNT=3

def setup_dist():
    """
    Set up a distributed process group for parallel and distributed computing
    Initializes a PyTorch distributed process group using environment variables
    and an MPI communicator to synchronize between different MPI ranks
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
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    return torch.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch state dictionary from a file path with minimized redundant fetches across MPI ranks.
    :param path: The file path to load the state dictionary from
    :param kwargs: Additional keyword arguments to pass to 'torch.load'
    :return:
        - The state dictionary loaded from the file
    """
    chunk_size=2**30 # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() ==0:
        with bf.BlobFile(path, "rb") as f:
            data =f.read()
        num_chunks=len(data)//chunk_size
        if len(data) % chunk_size:
            num_chunks+=1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0,len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i+chunk_size])
    else:
        num_chunks=MPI.COMM_WORLD.bcast(None)
        data=bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return torch.load(io.BytesIO(data), **kwargs)



def sync_params(params):
    """
    Synchronize a sequence of Tensors across MPI ranks from rank 0
    :param params:A sequence of PyTorch Tensors to synchronize
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p,0)

def _find_free_port():
    """
    Find and return a free port on the system by creating, binding, and closing a socket
    :return:
    -  int: The free port number found
    """
    try:
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()