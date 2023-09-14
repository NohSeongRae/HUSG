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

def load_mask(dir_name):
    mask_list = []

    file_list = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]

    for i in range(len(file_list)-1):
        image_path_png = file_list[i]
        image_path = os.path.join(dir_name, image_path_png)
        mask_image = Image.open(image_path)
        mask_numpy = np.array(mask_image, dtype=np.float32) / 255.0
        mask_tensor = torch.tensor(mask_numpy)
        # mask_tensor = mask_tensor.long()
        # print(mask_tensor.size())
        # print(mask_tensor)
        mask_list.append(mask_tensor)

    return mask_list

centroidmask = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask', 'centroidmask_sample')

# inverse_masks = load_mask(inversemask)
centriod_masks = load_mask(centroidmask)