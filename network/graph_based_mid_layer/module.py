import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(dim)

        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x1 = F.relu(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)

        x2 = F.relu(x1)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)

        return x + x2

class ResidualStack(nn.Module):
    def __init__(self, d_model, n_layer):
        super(ResidualStack, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([ResidualBlock(d_model)
                                     for _ in range(self.n_layer)])

    def forward(self, x):
        for i in range(self.n_layer):
            x = self.layers[i](x)
        return F.relu(x)