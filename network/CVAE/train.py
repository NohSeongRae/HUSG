import torch, os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from urban_dataset import UrbanGraphDataset, graph_transform, get_transform
from model import *
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch
import numpy as np
import random
import argparse
# from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import MultiStepLR
from time import gmtime, strftime
import shutil
import logging
from graph_util import read_train_yaml
from graph_trainer import train, validation
import yaml
import warnings
import dill as pickle
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from graph_trainer import Trainer

if __name__=="__main__":
    random.seed(42)
    # root=os.getcwd()

    dataset_path=r'Z:\iiixr-drive\Projects\2023_City_Team\4_CVAE'
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--pad_idx", type=int, default=2, help="Padding index for sequences.")
    parser.add_argument("--sos_idx", type=int, default=3, help="Padding index for sequences.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--d_street", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--d_unit", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_building", type=int, default=1, help="binary classification for building existence.")
    parser.add_argument("--n_boundary", type=int, default=200, help="Number of boundary or token.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    # parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--train_ratio", type=float, default=0.89, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Use checkpoint index.")
    parser.add_argument("--scheduler_step", type=int, default=20, help="Use checkpoint index.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--use_global_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_street_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_local_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="default_path", help="save dir path")

    args=parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)

    rank = args.local_rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')

    trainer=Trainer(args)
    trainer.train()