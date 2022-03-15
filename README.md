# Keyword-Spotting-Playground
AI Research into Spoken Keyword Spotting. 
Collection of PyTorch implementations of Spoken Keyword Spotting presented in research papers.
Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. 


# Implementations

## About DataSet
[Speech Commands DataSet](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) is a set of one-second .wav audio files, each containing a single spoken English word.
These words are from a small set of commands, and are spoken by a variety of different speakers.
The audio files are organized into folders based on the word they contain, and this dataset is designed to help train simple machine learning models.

## Installation
We use the Google Speech Commands Dataset (GSC) as the training data. By running the script, you can download the training data:

```bash
cd <ROOT>/dataset
bash download_gsc.sh
```

# Reference
1. https://github.com/hyperconnect/TC-ResNet
2. https://github.com/eriklindernoren/PyTorch-GAN

# Acknowledgement
Thanks for the contribution by [huangyz0918](https://github.com/huangyz0918)!