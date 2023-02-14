"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午6:27
@Author  : Yang "Jan" Xiao 
@Description : RNN
"""
import torch
import torch.nn as nn
from torchaudio.transforms import MFCC


class RNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512, n_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        batch_size, _, n_mfcc, _ = inputs.shape
        inputs = inputs.reshape(batch_size, -1, n_mfcc)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(inputs, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MFCC_RNN(nn.Module):
    def __init__(self, n_mfcc, sampling_rate, n_layers=1, hidden_size=512, num_classes=12):
        super(MFCC_RNN, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_classes = num_classes
        self.n_mfcc = n_mfcc  # feature length

        self.mfcc_layer = MFCC(sample_rate=self.sampling_rate, n_mfcc=self.n_mfcc, log_mels=True)
        self.rnn = RNN(self.n_mfcc, self.num_classes, hidden_size=hidden_size, n_layers=n_layers)

    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        logits = self.rnn(mel_sepctogram)
        return logits
