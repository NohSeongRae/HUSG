import math

import torch
import torch.nn as nn


# for Pytorch <1.7
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D or 3D convolution module
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using an
    Exponential Moving Average
    :param target_params:
    :param source_params:
    :param rate: EMA rate (closer to 1 means ema moving slower)
    :return:
    """
    for tar, src in zip(target_params, source_params):
        tar.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zeros out the parameters of a module and return it
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it
    :return:
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor, padding_mask):
    """
    Take the mean over all non-batch dimensions
    :param tensor:
    :param padding_mask:
    :return:
    """
    tensor = tensor * padding_mask.unsqueeze(1)
    tensor = tensor.mean(dim=list(range(1, len(tensor.shape)))) / torch.sum(padding_mask, dim=1)
    return tensor


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels
    :return: nn.Module for normalization
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Generate sinusoidal embeddings for the given timesteps. This is often used in transformers
    and other architectures to provide a sense of order in sequences.
    :param timesteps: A 1-D Tensor of N indices (integers), one per batch element
    :param dim:The dimensionality of the output embeddings
    :param max_period: Controls the minimum frequency of the embeddings, affecting periodicity and wavelength
    :return: A [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Utilize gradient checkpointing to evaluate a function without storing intermediate activations.
    This reduces memory usage but increase computational cost during backpropagation because the forward
    pass needs to be recomputed
    :param func: The function to be evaluated
    :param inputs: Inputs to be passed to 'func'
    :param params: Parameters that 'func' depends on, explicitly passed.
    :param flag: If False, gradient checkpointing is disabled.
    :return: The output of the evalueated 'func'
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        """
        Forward pass for the checkpointing function
        :param ctx: A context object that can be used to stash information for backward computation
        :param run_function: The function to run with checkpointing.
        :param length: The number of inputs to consider from args.
        :param args: The arguements to pass to 'run_function'
        :return: Output of 'run_function'
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """
        Backward pass for the checkpointing function
        The intermediate results from the forward pass are recomputed here
        to save memory, as they were not saved during the forward pass
        :param ctx: A context object containing information from the forward pass.
        :param output_grads: Gradients of the output to propagate back
        :return: Gradients with respect to the input parameters
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
