import copy
import functools
import os

import blobfile as bf
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

import dist_util, logger
from fp16_util import MixedPrecisionTrainer
from nn import update_ema
from resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, *, model, diffusion, data, batch_size, microbatch, lr, ema_rate, log_interval, save_interval,
                 resume_checkpoint,
                 use_fp16=False, fp16_scale_growth=1e-3, schedule_sampler=None, weight_decay=0.0, lr_anneal_steps=0,
                 analog_bit=None):
        pass

    def _load_and_sync_parameters(self):
        pass

    def _load_ema_parameters(self, rate):
        pass

    def _load_optimizer_state(self):
        pass

    def run_loop(self):
        pass

    def run_step(self, batch, cond):
        pass

    def forward_backward(self, batch, cond):
        pass

    def _update_ema(self):
        pass

    def _anneal_lr(self):
        pass

    def log_step(self):
        pass

    def save(self):
        pass


def parse_resume_step_from_filename(filename):
    pass


def get_blob_logdir():
    pass


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    pass

