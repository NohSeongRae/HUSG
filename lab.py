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

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
result_along_axis0 = np.cumprod(arr_2d, axis=0)
result_along_axis1 = np.cumprod(arr_2d, axis=1)

print("Cumprod along axis 0:")
print(result_along_axis0)

print("\nCumprod along axis 1:")
print(result_along_axis1)

