import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from network.image_based_mid_layer.encoder import ResnetBlock, BasicBlock1DConv
from network.image_based_mid_layer.decoder import DecoderBlock

nonlinearity = partial(F.relu, inplace=True)

class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out+x))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out+x))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out+x))
        out = x + dilate1_out + dilate2_out + dilate3_out
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        myresnet = ResnetBlock()
        layers = [3, 4, 6]
        basicBlock = BasicBlock1DConv
        self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
        self.encoder2 = myresnet._make_layer(basicBlock, 128, layers[1], stride=2)
        self.encoder3 = myresnet._make_layer(basicBlock, 256, layers[2], stride=2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        return e1, e2, e3

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

    def forward(self, e1, e2, e3):
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        return d1

class ImageNet(nn.Module):
    def __init__(self, n_mask):
        super().__init__()

        self.conv_main = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv_sub = nn.Conv2d(n_mask - 3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn_init = nn.BatchNorm2d(64)

        self.encoder = Encoder()
        self.dblock = DBlock(256)
        self.decoder = Decoder()

        self.conv_final = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        x_main = self.conv_main(x[:, :3])
        x_sub = self.conv_sub(x[:, 3:])

        x_sum = self.bn_init(x_main + x_sub)
        x_sum = nonlinearity(x_sum)

        e1, e2, e3 = self.encoder(x_sum)
        e3 = self.dblock(e3)
        d1 = self.decoder(e1, e2, e3)

        out = self.conv_final(d1)
        out = torch.sigmoid(out)

        return out