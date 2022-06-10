"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午10:50
@Author  : Yang "Jan" Xiao 
@Description : train_utils
"""
import torch_optimizer
from torch import optim

from networks.bcresnet import MFCC_BCResnet
from networks.tcresnet import MFCC_TCResnet
from networks.matchboxnet import MFCC_MatchboxNet


def select_optimizer(opt_name, lr, model, sched_name="cos"):
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    elif opt_name == "NovoGrad":
        opt = torch_optimizer.NovoGrad(model.parameters(), lr=0.05, betas=(0.95, 0.5), weight_decay=0.001)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")

    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=lr * 0.01
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, list(range(5, 26)), gamma=0.85
        )
    else:
        raise NotImplementedError(
            "Please select the sched_name [cos, anneal, multistep]"
        )

    return opt, scheduler


def select_model(model_name, total_class_num=None):
    # load the model.
    config = {
        "tcresnet8": [16, 24, 32, 48],
        "tcresnet14": [16, 24, 24, 32, 32, 48, 48]
    }

    if model_name == "tcresnet8":
        model = MFCC_TCResnet(bins=40, channels=config[model_name], channel_scale=1,
                              num_classes=total_class_num)
    elif model_name == "tcresnet14":
        model = MFCC_TCResnet(bins=40, channels=config[model_name], channel_scale=1,
                              num_classes=total_class_num)
    elif model_name == "bcresnet":
        model = MFCC_BCResnet(bins=40, channel_scale=1, num_classes=30)
    elif model_name == "bcresnet8":
        model = MFCC_BCResnet(bins=40, channel_scale=8, num_classes=30)
    elif model_name == "matchboxnet":
        model = MFCC_MatchboxNet(bins=64, B=3, R=2, n_channels=64, num_classes=30)
    else:
        model = None

    return model
