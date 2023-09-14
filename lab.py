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


x=torch.tensor([1,2,3,4])
print(x)
print(f"x shape: {x.size()}")
print(f'x, 0 :{torch.unsqueeze(x,0)}')
x_1=torch.unsqueeze(x, 1)

print(f'x, 1 :{x_1}')
print(f'x_1 shape: {x_1.size()}')
y=torch.tensor([[1,2],[3,4]])
print(f'y shape: {y.size()}')
y_0=torch.unsqueeze(y,0)
print(f'y, 0 :{y_0}')
print(f"y_0 shape: {y_0.size()}")
y_1=torch.unsqueeze(y,1)
print(f'y, 1 :{y_1}')
print(f"y_1 shape: {y_1.size()}")
y_2=torch.unsqueeze(y,2)
print(f'y, 2 :{y_2}')
print(f"y_2 shape: {y_2.size()}")
print(y)

