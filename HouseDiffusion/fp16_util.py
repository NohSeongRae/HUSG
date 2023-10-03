import numpy as np
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    pass


def convert_module_to_f32(l):
    pass


def make_master_params(param_groups_and_shapes):
    pass


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    pass


def master_params_to_model_params(param_groups_and_shapes, master_params):
    pass


def unflatten_master_params(param_group, master_param):
    pass


def get_param_groups_and_shapes(named_model_params):
    pass


def master_params_to_state_dict(model, param_groups_and_shapes, master_params, use_fp16):
    pass


def state_dict_to_master_params(model, state_dict, use_fp16):
    pass


def zero_master_grads(master_params):
    pass


def zero_grad(model_params):
    pass


def param_grad_or_zeros(param):
    pass


class MixedPrecisionTrainer:
    def __init__(self, *, model, use_fp16=False, fp16_scale_growth=1e-3, initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE):
        pass

    def zero_grad(self):
        pass

    def backward(self, loss: torch.Tensor):
        pass

    def optimize(self, opt: torch.optim.Optimizer):
        pass

    def _optimize_fp16(self, opt: torch.optim.Optimizer):
        pass

    def _optimize_normal(self, opt: torch.optim.Optimizer):
        pass

    def _compute_norms(self, grad_scale=1.0):
        pass

    def master_params_to_state_dict(self, master_params):
        pass

    def state_dict_to_master_params(self, state_dict):
        pass


def check_overflow(value):
    pass
