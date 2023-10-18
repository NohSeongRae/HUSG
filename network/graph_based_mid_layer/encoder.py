import torch
import torch.nn as nn
import torch.nn.functional as F
from network.graph_based_mid_layer.module import ResidualStack

class Encoder(nn.Module):
    def __init__(self, d_in, d_model, n_res_layer):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(d_in, d_model // 2,
                               kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model // 2, d_model,
                               kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(d_model, d_model,
                               kernel_size=3, stride=1, padding=1)
        self.res_stack = ResidualStack(d_model=d_model, n_layer=n_res_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        return self.res_stack(x)