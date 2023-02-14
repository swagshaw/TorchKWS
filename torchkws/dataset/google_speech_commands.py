import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
class SpeechCommandsDataset(Dataset):
    def __init__(self, root_path, subset='train', transform=None):
        self.subset = subset
        self.transform = transform

        self.file_names = []
        self.labels = []

        for label, name in enumerate(os.listdir(os.path.join(root_path, subset))):
            folder_path = os.path.join(root_path, subset, name)
            for file_name in os.listdir(folder_path):
                self.file_names.append(os.path.join(folder_path, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label