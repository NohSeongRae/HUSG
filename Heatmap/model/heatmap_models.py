import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(f'runs/loc_experiment/{current_time}')


def Conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PolygonResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, num_input_channels=17, use_fc=False):
        self.inplanes = 64
        super(PolygonResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, out_channels=64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 28, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight initializing - variant of "He initialization"
        for m in self.modules():
            if isinstance(m,
                          nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(
                    2. / n))  # Initialize the weights of the convolutional layer using a normal distribution.

            elif isinstance(m,
                            nn.BatchNorm2d):
                m.weight.data.fill_(1)

                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x


def resnet34(pretrained=False, **kwargs):
    model = PolygonResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class DownConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=2, kernel_size=4, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.upsample(x, mode='nearest', scale_factor=2)
        return self.act(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.upsample(x, mode='nearest', scale_factor=2)
        return self.act(self.bn(self.conv(x)))


class Model(nn.Module):
    def __init__(self, num_classes, num_input_channels):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            resnet34(num_input_channels=num_input_channels),
            nn.Dropout(p=0.1),
            UpConvBlock(512, 256),
            UpConvBlock(256, 128),
            UpConvBlock(128, 64),
            UpConvBlock(64, 32),
            UpConvBlock(32, 16),
            UpConvBlock(16, 8),
            nn.Dropout(p=0.1),
            nn.Conv2d(8, num_classes, 1, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
