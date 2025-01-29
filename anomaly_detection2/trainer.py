from torch import nn
import torch
from anomaly_detection2 import models
from torch.utils.data import DataLoader
class BaseTrainer():
    def __init__(self, model: models.Autoencoder ,
                  trainloader: DataLoader, optimizer, criterion, device: str):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model = self.model.to(device)

    def train(self, num_epochs = 20):
        """train the model with the given torch objects

        Args:
            num_epochs (int, optional): Num epochs. Defaults to 20.
        """

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for batch in self.trainloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.trainloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

        return self.model