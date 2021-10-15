"""Helpers to train with 16-bit precision."""

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def convert_module_to_f16(l_):
    """convert primitive modules to float16."""
    if isinstance(l_, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l_.weight.data = l_.weight.data.half()
        l_.bias.data = l_.bias.data.half()


def convert_module_to_f32(l_):
    """Convert primitive modules to float32, undoing
    convert_module_to_f16()."""
    if isinstance(l_, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l_.weight.data = l_.weight.data.float()
        l_.bias.data = l_.bias.data.float()


def make_master_params(model_params):
    """Copy model parameters into a (differently-shaped) list of full-precision
    parameters."""
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params])
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """Copy the gradients from the model parameters into the master parameters
    from make_master_params()."""
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params])


def master_params_to_model_params(model_params, master_params):
    """copy the master parameter data back into the model parameters."""
    # without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for param, master_param in zip(
            model_params, unflatten_master_params(model_params,
                                                  master_params)):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """unflatten the master parameters to look like model_params."""
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def zero_grad(model_params):
    for param in model_params:
        # taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#optimizer.add_param_group  # noqa
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
