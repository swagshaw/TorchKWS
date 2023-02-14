# TorchKWS
TorchKWS is a collection of PyTorch implementations of spoken keyword spotting models that have been presented in research papers. It includes implementations of various neural network architectures, as well as utilities for data preprocessing, training, and evaluation.
## Installation
To use TorchKWS, you'll need to have PyTorch and other required dependencies installed. You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
## Dataset
To use TorchKWS, you'll need to have a dataset of speech recordings and corresponding keyword labels. One example dataset you can use is the [Google Speech Commands dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html), which consists of 65,000 one-second long utterances of 30 short words, by thousands of different people.

To download and preprocess the Google Speech Commands dataset, you can use the `download_google_speech_commands.py` script in the data directory:

This will download the dataset and preprocess it into WAV files and CSV files containing the file paths and labels for the training, validation, and test sets.

## Usage
To train a keyword spotting model using TorchKWS, you can use the `train.py` script. This script takes a configuration file as input, which specifies the hyperparameters for the model, training, and evaluation.

Here's an example of how to train a model using the configuration file `config.yaml`:

```bash
python train.py --config_path config.yaml
```

You can modify the hyperparameters in the `config.yaml` file to experiment with different model architectures and training strategies.

## Package Structure
The TorchKWS repository is organized into several packages:

- `data`: Contains utilities for loading and preprocessing datasets, as well as scripts for downloading and preprocessing specific datasets.
- `models`: Contains implementations of various neural network architectures for keyword spotting.
- `training`: Contains utilities for training and evaluating models, as well as the `train.py` script for training models using a specified configuration file.
- `utils`: Contains miscellaneous utility functions.

## Contributions
Contributions to TorchKWS are welcome! If you'd like to contribute, please submit a pull request. We also welcome suggestions for new models to implement or new features to add to the repository.

## License
TorchKWS is released under the [MIT License](https://github.com/example/torchkws/blob/main/LICENSE).
