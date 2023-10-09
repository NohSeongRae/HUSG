import numpy as np
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert the weights and biases of convolutional layers to float16
    :param l:
    :return:
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert the weights and biases of convolutional layers back to float32
    :param l:
    :return:
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Create master parameters (float32) from the model parameters.
    Used to store high-precision copies of the model parameters.
    :param param_groups_and_shapes:
    :return:
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy gradient from the model parameters to the master parameters.
    Gradients computed in reduced precision are copied to master parameters for accurate update
    :param param_groups_and_shapes:
    :param master_params:
    :return:
    """
    for master_params, (param_group, shape) in zip(
            master_params, param_groups_and_shapes
    ):
        master_params.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copies the data from master parameters (high precision) to model parameters (low precision).
    :param param_groups_and_shapes:Contains information about parameter groups and their shapes
    :param master_params:High precision parameters used for computations
    """
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    """
    Unflattens the master parameter to their original shapes.
    :param param_group: Group of parameters.
    :param master_param: Flatten master parameter
    :return:
        Unflatten master parameter
    """
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    """
    Seperates parameters into scalar/vector and matrix parameters,
    and determines their shapes.
    :param named_model_params: Model parameters with names
    :return:
        Group of scalar/vector and matrix parameters
    """
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1)
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(model, param_groups_and_shapes, master_params, use_fp16):
    """
    Copies the master parameters to the model's state dictionary
    :param model:The model whose state dictionary will be updated
    :param param_groups_and_shapes: Contains information about parameter groups and their shapes
    :param master_params: High precision parameters used for computations.
    :param use_fp16: A flag indicating whether to use mixed precision or not
    :return:
        Updated state dictionary of the model
    """
    if use_fp16:
        state_dict = model.state_dict()
        for master_params, (param_group, _) in zip(
                master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                    master_params, unflatten_master_params(param_group, master_params.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    """
    Retrieves the master parameters from the state dictionary of the model
    :param model: The model whose parameters will be retrieved
    :param state_dict: State dictionary containing the parameters
    :param use_fp16: A flag indicating whether to use mixed precision or not
    :return:
        The master parameters
    """
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        params_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(params_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    """
    Sets the gradient of each master parameter to None
    :param master_params: High precision parameters used for computations
    """
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    """
    Zeros the gradients of the model parameters
    """
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    """
    Returns the gradient of a parameter, or zeros if the gradient is None
    :param param: The parameter whose gradient is to be returned.
    :return: Gradient of the parameter or zeros
    """
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return torch.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(self, *, model, use_fp16=False, fp16_scale_growth=1e-3, initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE):
        """
        Initialize the MixedPrecisionTrainer
        :param model: The model to be trained
        :param use_fp16: Boolean, whether to use mixed-precision training.
        :param fp16_scale_growth: Scale growth for mixed precision training
        :param initial_lg_loss_scale: Initial logarithmic loss scale
        """
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        """
        Zero out the gradients of all model parameters
        """
        zero_grad(self.model_params)

    def backward(self, loss: torch.Tensor):
        """
        Perform the backward pass through the model
        :param loss: Loss tensor to backpropagate
        """
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: torch.optim.Optimizer):
        """
        Perform the optimization step
        :param opt: The optimizer to be used for parameter updates
        """
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: torch.optim.Optimizer):
        """
        Perform the optimization step using mixed precision
        :param opt: The optimizer to be used for parameter updates
        """
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale = -1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale = self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: torch.optim.Optimizer):
        """
        Perform the optimization step without using mixed precision
        :param opt: The optimizer to be used for parameter updates
        """
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        """
        Compute the norms of gradients and parameters
        :param grad_scale: The scale to be used for the gradient norms.
        """
        grad_norm=0.0
        param_norm=0.0
        for p in self.master_params:
            with torch.no_grad():
                param_norm+=torch.norm(p, p=2, dtype=torch.float32).item()**2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item()**2
        return np.sqrt(grad_norm)/grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        """
        Convert master parameters to a state dictionary for checkpointing
        :param master_params: The master parameters to be converted
        """
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        """
        Convert a state dictionary back to master parameters for training
        :param state_dict: The state dictionary to be converted
        """
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
