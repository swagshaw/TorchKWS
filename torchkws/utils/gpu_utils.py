"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午5:43
@Author  : Yang "Jan" Xiao 
@Description : utils
"""
import numpy as np
import torch


class ColorEnum:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def print_color(color, *msgs):
    print(color, *msgs, ColorEnum.END)


def readlines(datapath):
    # Read all the lines in a text file and return as a list
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def padding(batch, seq_len):
    if len(batch[0][0]) < seq_len:
        m = torch.nn.ConstantPad1d((0, seq_len - len(batch[0][0])), 0)
        batch = m(batch)
    return batch


def parameter_number(model):
    num_params = 0
    for tensor in list(model.parameters()):
        tensor = tensor.view(-1)
        num_params += len(tensor)
    return num_params


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print_color(ColorEnum.YELLOW, "There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print_color(ColorEnum.YELLOW,
                    "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                        n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids
