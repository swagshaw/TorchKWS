"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/6/4 下午4:26
@Author  : Yang "Jan" Xiao 
@Description :
reference:https://github.com/dominickrei/MatchboxNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC
from utils.utils import padding

class TCSConv(nn.Module):
    '''
    An implementation of Time-channel Seperable Convolution
    **Arguments**
    in_channels : int
      The number of input channels to the layers
    out_channels : int
      The requested number of output channels of the layers
    kernel_size : int
      The size of the convolution kernel
    Example
    -------
    >>> inputs = torch.randn(1, 64, 400)
    >>> tcs_layer = TCSConv(64, 128, 11)
    >>> features = tcs_layer(inputs)
    >>> features.shape
    torch.Size([1, 128, 400])
    '''

    def __init__(self, in_channels, out_channels, kernel_size):
        super(TCSConv, self).__init__()

        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels,
                                        padding='same')  # effectively performing a depthwise convolution
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels,
                                        kernel_size=1)  # effectively performing a pointwise convolution

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class SubBlock(nn.Module):
    '''
    An implementation of a sub-block that is repeated R times
    **Arguments**
    in_channels : int
      The number of input channels to the layers
    out_channels : int
      The requested number of output channels of the layers
    kernel_size : int
      The size of the convolution kernel

    residual : None or torch.Tensor
      Only applicable for the final sub-block. If not None, will add 'residual' after batchnorm layer
    Example
    -------
    >>> inputs = torch.randn(1, 128, 600)

    >>> subblock = SubBlock(128, 64, 13)
    >>> outputs = subblock(inputs)
    >>> outputs.shape
    torch.Size([1, 64, 600])
    '''

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SubBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.tcs_conv = TCSConv(self.in_channels, self.out_channels, self.kernel_size)
        self.bnorm = nn.BatchNorm1d(self.out_channels)
        self.dropout = nn.Dropout()

    def forward(self, x, residual=None):
        x = self.tcs_conv(x)
        x = self.bnorm(x)

        # apply the residual if passed
        if residual is not None:
            x = x + residual

        x = F.relu(x)
        x = self.dropout(x)

        return x


class MainBlock(nn.Module):
    '''
    An implementation of the residual block containing R repeating sub-blocks
    **Arguments**
    in_channels : int
      The number of input channels to the residual block
    out_channels : int
      The requested number of output channels of the sub-blocks
    kernel_size : int
      The size of the convolution kernel
    R : int
      The number of repeating sub-blocks contained within this residual block

    residual : None or torch.Tensor
      Only applicable for the final sub-block. If not None, will add 'residual' after batchnorm layer
    Example
    -------
    >>> inputs = torch.randn(1, 128, 300)

    >>> block = MainBlock(128, 64, 13, 3)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([1, 64, 300])
    '''

    def __init__(self, in_channels, out_channels, kernel_size, R=1):
        super(MainBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.residual_pointwise = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm1d(self.out_channels)

        self.sub_blocks = nn.ModuleList()

        # Initial sub-block. If this is MainBlock 1, our input will be 128 channels which may not necessarily == out_channels
        self.sub_blocks.append(
            SubBlock(self.in_channels, self.out_channels, self.kernel_size)
        )

        # Other sub-blocks. Output of all of these blocks will be the same
        for i in range(R - 1):
            self.sub_blocks.append(
                SubBlock(self.out_channels, self.out_channels, self.kernel_size)
            )

    def forward(self, x):
        residual = self.residual_pointwise(x)
        residual = self.residual_batchnorm(residual)

        for i, layer in enumerate(self.sub_blocks):
            if (i + 1) == len(self.sub_blocks):  # compute the residual in the final sub-block
                x = layer(x, residual)
            else:
                x = layer(x)

        return x


class MatchboxNet(nn.Module):
    '''
    An implementation of MatchboxNet (https://arxiv.org/abs/2004.08531)
    The input is expected to be 64 channel MFCC features
    **Arguments**
    B : int
      The number of residual blocks in the model
    R : int
      The number of sub-blocks within each residual block
    C : int
      The size of the output channels within a sub-block
    kernel_sizes : None or list
      If None, kernel sizes will be assigned to values used in the paper. Otherwise kernel_sizes will be used
      len(kernel_sizes) must equal the number of blocks (B)
    NUM_CLASSES : int
      The number of classes in the dataset (i.e. number of keywords.) Defaults to 30 to match the Google Speech Commands Dataset
    Example
    -------
    >>> inputs = torch.randn(1, 64, 500)

    >>> model = MatchboxNet(B=3, R=2, C=64,bins=64, NUM_CLASSES=30)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([1, 30])
    '''

    def __init__(self, B, R, C, bins=64, kernel_sizes=None, NUM_CLASSES=30):
        super(MatchboxNet, self).__init__()
        if not kernel_sizes:
            kernel_sizes = [k * 2 + 11 for k in range(1, 5 + 1)]  # incrementing kernel size by 2 starting at 13

        # the prologue layers
        self.prologue_conv1 = nn.Conv1d(bins, 128, kernel_size=11, stride=2)
        self.prologue_bnorm1 = nn.BatchNorm1d(128)

        # the intermediate blocks
        self.blocks = nn.ModuleList()

        self.blocks.append(
            MainBlock(128, C, kernel_sizes[0], R=R)
        )

        for i in range(1, B):
            self.blocks.append(
                MainBlock(C, C, kernel_size=kernel_sizes[i], R=R)
            )

        # the epilogue layers
        self.epilogue_conv1 = nn.Conv1d(C, 128, kernel_size=29, dilation=2)
        self.epilogue_bnorm1 = nn.BatchNorm1d(128)

        self.epilogue_conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.epilogue_bnorm2 = nn.BatchNorm1d(128)

        self.epilogue_conv3 = nn.Conv1d(128, NUM_CLASSES, kernel_size=1)

        # Pool the timesteps into a single dimension using simple average pooling
        self.epilogue_adaptivepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # prologue block
        x = self.prologue_conv1(x)
        x = self.prologue_bnorm1(x)
        x = F.relu(x)

        # intermediate blocks
        for layer in self.blocks:
            x = layer(x)

        # epilogue blocks
        x = self.epilogue_conv1(x)
        x = self.epilogue_bnorm1(x)

        x = self.epilogue_conv2(x)
        x = self.epilogue_bnorm2(x)

        x = self.epilogue_conv3(x)
        x = self.epilogue_adaptivepool(x)
        x = x.squeeze(2)  # (N, 30, 1) > (N, 30)
        x = F.softmax(x, dim=1)  # softmax across classes and not batch

        return x


class MFCC_MatchboxNet(nn.Module):
    def __init__(self, bins: int, B: int, R: int, n_channels, kernel_sizes=None, num_classes=12):
        super(MFCC_MatchboxNet, self).__init__()
        self.sampling_rate = 16000
        self.bins = bins
        self.num_classes = num_classes
        self.mfcc_layer = MFCC(sample_rate=self.sampling_rate, n_mfcc=self.bins, log_mels=True)
        self.matchboxnet = MatchboxNet(B, R, n_channels, bins=self.bins, kernel_sizes=kernel_sizes,
                                       NUM_CLASSES=num_classes)

    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        mel_sepctogram = mel_sepctogram.squeeze(1)
        mel_sepctogram = padding(mel_sepctogram, 128)
        logits = self.matchboxnet(mel_sepctogram)
        return logits


if __name__ == "__main__":
    inputs = torch.randn(128, 64, 50)
    inputs = padding(inputs, 128)
    model = MatchboxNet(B=3, R=2, C=64, bins=64, NUM_CLASSES=30)
    outputs = model(inputs)
    print(outputs.shape)
