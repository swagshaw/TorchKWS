# TorchKWS
AI Research into Spoken Keyword Spotting. 
Collection of PyTorch implementations of Spoken Keyword Spotting presented in research papers.
Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. 

# Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Temporal Convolution Resnet(TC-ResNet)](#temporal-convolution-resnet)
    + [Broadcasting Residual Network(BC-ResNet)](#broadcasting-residual-network)
    + [MatchboxNet](#matchboxnet)
    + [ConvMixer](#convmixer)
    + [Keyword Transformer(KWT)](#kwt)


# Implementations
## About DataSet
[Speech Commands DataSet](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) is a set of one-second .wav audio files, each containing a single spoken English word.
These words are from a small set of commands, and are spoken by a variety of different speakers.
The audio files are organized into folders based on the word they contain, and this dataset is designed to help train simple machine learning models.

## Installation
We use the Google Speech Commands Dataset (GSC) as the training data. By running the script, you can download the training data:

```python
cd <ROOT>/dataset
python process_speech_commands_data.py \
    --data_root=<absolute path to where the data should be stored> \
    --data_version=<either 1 or 2, indicating version of the dataset>\
    --class_split=<either "all" or "sub", indicates whether all 30/35 classes should be used, or the 10+2 split should be used> \
    --rebalance \
    --log
```

## Temporal Convolution Resnet
_Temporal Convolution for Real-time Keyword Spotting on Mobile Devices_
[[Paper]](https://arxiv.org/abs/1904.03814) [[Code]](networks/tcresnet.py)

## Broadcasting Residual Network
_Broadcasted Residual Learning for Efficient Keyword Spotting_
[[Paper]](https://arxiv.org/abs/2106.04140) [[Code]](networks/bcresnet.py)

## MatchboxNet
_MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition_
[[Paper]](https://arxiv.org/abs/2004.08531) [[Code]](networks/matchboxnet.py)

## ConvMixer
_ConvMixer: Feature Interactive Convolution with Curriculum Learning for Small Footprint and Noisy Far-field Keyword Spotting_
[[Paper]](https://arxiv.org/abs/2201.05863) [[Code]](networks/convmixer.py)

## KWT
_Keyword transformer: A self-attention model for keyword spotting_
[[Paper]](https://arxiv.org/abs/2104.00769) [[Code]](network/kwt.py)

# Reference
1. https://github.com/hyperconnect/TC-ResNet
2. https://github.com/huangyz0918/kws-continual-learning
3. https://github.com/eriklindernoren/PyTorch-GAN
4. https://github.com/roman-vygon/BCResNet
5. https://github.com/dominickrei/MatchboxNet
6. https://github.com/dianwen-ng/Keyword-Spotting-ConvMixer
