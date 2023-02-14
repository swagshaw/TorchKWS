from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, MelSpectrogram, Normalize, ToTensor

from torchkws.dataset.google_speech_commands import SpeechCommandsDataset
import torch.utils.data as data
from torchkws.utils import ToMelSpectrogram, ToTensor, Normalize


def create_data_loader(config, set_name):
    # Create the data transforms
    transforms = data.Compose([
        ToMelSpectrogram(n_mels=config['data']['num_mels'], n_fft=config['data']['num_fft'],
                         hop_length=config['data']['hop_length']),
        Normalize(),
        ToTensor()
    ])

    dataset = SpeechCommandsDataset(config['data']['root_dir'], subset=set_name, transform=transforms)

    # Split the dataset into train, validation, and test sets
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = int(config['data']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for each set
    train_loader = DataLoader(train_set, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['data']['batch_size'])
    test_loader = DataLoader(test_set, batch_size=config['data']['batch_size'])

    return train_loader, val_loader, test_loader
