import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, device, optimizer, scheduler, train_loader, val_loader, test_loader, num_epochs, checkpoint_dir, log_dir):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

    def train(self):
        train_losses = []
        val_losses = []
        test_losses = []
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []

        for epoch in range(self.num_epochs):
            # Set the model to training mode
            self.model.train()

            train_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_accuracy = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Set the model to evaluation mode
            self.model.eval()

            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0

                for i, data in enumerate(self.val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                val_accuracy = 100. * correct / total
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                test_loss = 0.0
                correct = 0
                total = 0

                for i, data in enumerate(self.test_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                test_accuracy = 100. * correct / total
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

            self.scheduler.step()

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

            # Save the model checkpoint
            torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/