import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from data_refine import load_mask


# 데이터셋
class BuildingDataset(Dataset):
    def __init__(self, boundary_masks, inside_masks, building_masks, boundarybuilding_masks, inverse_masks):
        self.boundary_masks = boundary_masks
        self.inside_masks = inside_masks
        self.building_masks = building_masks
        self.boundarybuilding_masks = boundarybuilding_masks
        self.inverse_masks = inverse_masks

    def __len__(self):
        return len(self.boundary_masks)

    def __getitem__(self, idx):
        boundary = self.boundary_masks[idx].unsqueeze(0)
        inside = self.inside_masks[idx].unsqueeze(0)
        building = self.building_masks[idx].unsqueeze(0)
        boundarybuilding = self.boundarybuilding_masks[idx].unsqueeze(0)
        inverse_masks = self.inverse_masks[idx].unsqueeze(0)

        x = torch.cat([boundary, inside, building, boundarybuilding, inverse_masks], dim=0)
        y = boundarybuilding

        return x, y


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

        out = self.con1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample

        out += residual
        out = self.relu(out)

        return out


class PolygonResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, num_input_channels=17, use_fc=False):
        self.inplanes = 64
        super(PolygonResNet, self).__init__()

        # base_model = resnet18(pretrained=True)
        # self.features = nn.Sequential(*list(base_model.children())[:-2]) # pooling layer, FC layer 제거
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

        # weight initializing - variant of "He initialization"
        for m in self.modules():  # Loop over all the modules (layers) in the current model
            if isinstance(m,
                          nn.Conv2d):  # Check if the current module is an instance of Conv2d (i.e., a 2D convolutional layer)
                n = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels  # Calculate 'n', which is the product of the kernel width, kernel height, and the number of output channels
                # This is often used for weight initialization purposes
                m.weight.data.normal_(0, math.sqrt(
                    2. / n))  # Initialize the weights of the convolutional layer using a normal distribution.
                # The mean of this distribution is 0, and the standard deviation is set as sqrt(2/n).
                # This is a variant of the 'He initialization' which is common for layers before ReLU activations.
            elif isinstance(m,
                            nn.BatchNorm2d):  # Check if the current module is an instance of BatchNorm2d (i.e., a 2D batch normalization layer)
                m.weight.data.fill_(1)  # Set the weights (also called gamma) of the batch normalization layer to 1.
                # This essentially means that, initially, the batch normalization won't scale the output.
                m.bias.data.zero_()  # Set the bias (also called beta) of the batch normalization layer to 0.
                # This means that, initially, the batch normalization won't shift the output.

        # resnet에 up-sampling 레이어 추가
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 1, kernel_size=1)
        # )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.explansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.explansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.explansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.explansion
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


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_checkpoint(model, filename):
    """Load a model checkpoint."""
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")

def ensuredir(dirname):
    """Ensure a directory exists"""
    if not os.path.exists(dirname):
            os.makedirs(dirname)


boundarybuildingmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarybuildingmask')
boundarymask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'boundarymask')
buildingmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'buildingmask')
insidemask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'insidemask')
inversemask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'inversemask')

boundarybuilding_masks = load_mask(boundarybuildingmask)
boundary_masks = load_mask(boundarymask)
building_masks = load_mask(buildingmask)
inside_masks = load_mask(insidemask)
inverse_masks = load_mask(inversemask)

dataset_size = len(boundarybuilding_masks)
train_size = int(dataset_size * 0.8)
test_size = int(dataset_size * 0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
    parser.add_argument('--batch-size', type=int, default=16, metavar='S')
    # parser.add_argument('--data-folder', type=str, default="bedroom_6x6", metavar='S')
    parser.add_argument('--num-workers', type=int, default=6, metavar='N')
    parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
    parser.add_argument('--train-size', type=int, default=6000, metavar='N')
    parser.add_argument('--save-dir', type=str, default="loc_test", metavar='S')
    parser.add_argument('--ablation', type=str, default=None, metavar='S')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N')
    parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
    parser.add_argument('--centroid-weight', type=float, default=10, metavar="N")
    args = parser.parse_args()

    save_dir = args.save_dir
    ensuredir(save_dir)

    num_categories = 2  # building이 있을 수 있는 곳, 있을 수 없는 곳
    num_input_channels = num_categories + 8  # WHY?
    logfile = open(f"{save_dir}/log_location.txt", 'w')


    def LOG(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()


    LOG('Building model...')
    model = PolygonResNet(num_classes=num_categories + 1, num_input_channels=num_input_channels)  # WHY category+1?

    weight = [args.centroid_weight for i in range(num_categories + 1)]
    weight[0] = 1
    print(weight)

    weight = torch.from_numpy(np.asarray(weight)).float().cuda()
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    softmax = nn.Softmax()

    LOG('Converting to CUDA')
    model.cuda()
    cross_entropy.cuda()

    LOG('Building dataset...')
    train_dataset = BuildingDataset(
        boundary_masks=boundarybuilding_masks[:train_size],
        inside_masks=inside_masks[:train_size],
        building_masks=building_masks[:train_size],
        boundarybuilding_masks=boundarybuilding_masks[:train_size],
        inverse_masks=inverse_masks[:train_size]
    )

    LOG('Building data loader...')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    LOG('Building optimizer...')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)

    if args.last_epoch < 0:
        load = False
        starting_epoch = 0
    else:
        load = True
        last_epoch = args.last_epoch

    if load:
        LOG('Loading saved models...')
        model.load_state_dict(torch.load(f"{save_dir}/location_{last_epoch}.pt"))
        optimizer.load_state_dict(torch.load(f"{save_dir}/location_optim_backup.pt"))
        starting_epoch = last_epoch + 1

    current_epoch = starting_epoch
    num_seen = 0

    model.train()
    LOG(f'===================================== Epoch {current_epoch}=====================================')


    def train():
        global num_seen, current_epoch
        for batch_idx, (data, target) \
                in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = cross_entropy(output, target)
            print(loss)

            loss.backward()
            optimizer.step()

            num_seen += args.batch_size
            if num_seen % 800 == 0:
                LOG(f'Examples{num_seen}/10000')
            if num_seen % 10000 == 0:
                LOG('Validating')
                num_seen = 0
                current_epoch += 1
                LOG(f'===================================== Epoch {current_epoch}=====================================')
                if current_epoch % 10 == 0:
                    torch.save(model.state_dict(), f"{save_dir}/location_{current_epoch}.pt")
                    torch.save(optimizer.state_dict(), f"{save_dir}/location_optim_backup.pt")


    while True:
        train()
