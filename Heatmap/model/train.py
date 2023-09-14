import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

city_name = "philadelphia"

# 데이터셋
class PolygonDataset(Dataset):
    def __init__(self, boundary_masks, inside_masks, polygon_masks):
        self.boundary_masks = boundary_masks
        self.inside_masks = inside_masks
        self.polygon_masks = polygon_masks

    def __len__(self):
        return len(self.boundary_masks)

    def __getitem__(self, idx):
        boundary = self.boundary_masks[idx].unsqueeze(0)
        inside = self.inside_masks[idx].unsqueeze(0)
        polygon = self.polygon_masks[idx].unsqueeze(0)

        x = torch.cat([boundary, inside, polygon], dim=0)
        y = polygon

        return x, y


class PolygonResNet(nn.Module):
    def __init__(self):
        super(PolygonResNet, self).__init__()

        base_model = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2]) # pooling layer, FC layer 제거

        # resnet에 up-sampling 레이어 추가
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_checkpoint(model, filename):
    """Load a model checkpoint."""
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=200, save_path=None):
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)


if __name__ == "__main__":