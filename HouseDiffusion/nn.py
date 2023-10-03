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
    """
    Create sinusoidal timestep embeddings
    :param timesteps: 1-D Tensor of N indices, one per batch element
    :param dim: the dimension of the output
    :param max_period:controls the minimum frequency of the embeddings
    :return:an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    pass


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fun_function, length, *args):
        pass

    @staticmethod
    def backward(ctx, *output_grads):
        pass
