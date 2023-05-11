import os
from argparse import ArgumentParser

"""
original: src/parser/base
"""
def add_misc_options(parser):
    """
    기타 옵션들
    :param parser: Argparser
    :return: None
    """
    group = parser.add_argument_group('Miscellaneous options')
    group.add_argument("--expname", default="exps", help="general directory to this experiments")
    group.add_argument("--folder", help="directory name to save models")


def add_cuda_options(parser):
    """
    Cuda option, default=True
    :param parser: Argparser
    :return: None
    """
    group = parser.add_argument_group('Cuda options')
    group.add_argument("--cuda", dest='cuda', action='store_true', help="if we want to try to use gpu")
    group.add_argument("--cpu", dest='cuda', action='store_false', help="if we wnat to use cpu")
    group.add_argument("--device", default=None, type=int, help="# of gpu device")

    group.set_defaults(cuda=True)


def adding_cuda(parameters):
    """
    set cuda device
    :param parameters: experiment settings
    :return: None
    """
    import torch
    if parameters["cuda"] and torch.cuda.is_available():
        if parameters.get('device') is not None:
            parameters["device"] = torch.device(f"cuda: {parameters['device']}")
        else:
            parameters["device"] = torch.device("cuda")
    else:
        parameters["device"] = torch.device("cpu")
