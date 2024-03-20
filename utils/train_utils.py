"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午10:50
@Author  : Yang "Jan" Xiao 
@Description : train_utils
"""
import torch
import torch_optimizer
from torch import optim
from torch import nn
from networks.bcresnet import BCResNet
from networks.tcresnet import TCResNet
from networks.matchboxnet import MatchboxNet
from networks.kwt import kwt_from_name
from networks.convmixer import KWSConvMixer
from torchaudio.transforms import MFCC
class MFCC_KWS_Model(nn.Module):
    def __init__(self, model) -> None:
        super(MFCC_KWS_Model,self).__init__()
        self.mfcc = MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64, "f_min": 20, "f_max": 8000},
        )
        self.model = model
    def forward(self, x):
        x = self.mfcc(x)
        x = self.model(x)
        return x


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

    if model_name == "tcresnet8" or model_name == "tcresnet14":
        model = MFCC_KWS_Model(TCResNet(bins=40, n_channels=[int(cha * 1) for cha in config[model_name]],
                              n_class=total_class_num))
    elif "bcresnet" in model_name:
        scale = int(model_name[-1])
        model = MFCC_KWS_Model(BCResNet(n_class=total_class_num, scale=scale))
    elif "matchboxnet" in model_name:
        b, r, c = model_name.split("_")[1:]
        model = MFCC_KWS_Model(MatchboxNet(B=int(b), R=int(r), C=int(c), bins=40, kernel_sizes=None,num_classes=total_class_num))
    elif "kwt" in model_name:
        model = MFCC_KWS_Model(kwt_from_name(model_name, total_class_num))
    elif "convmixer" in model_name:
        model = MFCC_KWS_Model(KWSConvMixer(input_size=[101, 40],num_classes=total_class_num))
    else:
        model = None
    print(model)
    return model


if __name__ == "__main__":
    inputs = torch.randn(8, 1, 16000)
    # inputs = padding(inputs, 128)
    model = select_model("bcresnet2", 15)
    outputs = model(inputs)
    print(outputs.shape)
    print('num parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

