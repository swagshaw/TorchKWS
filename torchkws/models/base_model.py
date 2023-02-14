import torch.nn as nn


class KWSModel(nn.Module):
    def __init__(self, num_classes):
        super(KWSModel, self).__init__()
        self.num_classes = num_classes

        # Define your model layers here
        # Example:
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Define your forward pass here
        # Example:
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x