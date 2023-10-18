import torch.nn as nn
import torch.nn.functional as F
from network.graph_based_mid_layer.module import ResidualStack

class Decoder(nn.Module):
    def __init__(self, d_out, d_model, n_res_layer):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(d_model, d_model,
                               kernel_size=3, stride=1, padding=1)
        self.res_stack = ResidualStack(d_model=d_model, n_layer=n_res_layer)
        self.conv2 = nn.ConvTranspose2d(d_model, d_model // 2,
                                        kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(d_model // 2, d_out,
                                        kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.res_stack(x)

        x = self.conv2(x)
        x = F.relu(x)

        return self.conv3(x)