import math

import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        pass


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        pass


def conv_nd(dims, *args, **kwargs):
    pass


def linear(*args, **kwargs):
    pass


def avg_pool_nd(dims, *args, **kwargs):
    pass


def update_ema(target_params, source_params, rate=0.99):
    pass


def zero_module(module):
    pass


def scale_module(module, scale):
    pass


def mean_flat(tensor, padding_mask):
    pass


def normalization(channels):
    pass


def timestep_embedding(timesteps, dim, max_period=10000):
    pass


def checkpoint(func, inputs, params, flag):
    pass


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fun_function, length, *args):
        pass

    @staticmethod
    def backward(ctx, *output_grads):
        pass
