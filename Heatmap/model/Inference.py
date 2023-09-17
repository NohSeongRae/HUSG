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
from tqdm import tqdm
import paths
from heatmap_models import Model
from data_utils import get_datasets_and_loaders


def ensuredir(dirname):
    """Ensure a directory exists"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def tensor_to_image(tensor, temperature_pixel=0.8):
    probs = torch.nn.functional.softmax(tensor, dim=2)
    upsampled_tensor = F.interpolate(probs, size=(256, 256), mode='bilinear', align_corners=True)
    location_map = upsampled_tensor.cpu()
    location_map = location_map**(1/temperature_pixel)
    location_map = location_map / location_map.sum()
    # _, predictions = probs.max(2)
    # img = np.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=np.uint8)
    # img[predictions == 0] = [0, 0, 0]
    # img[predictions == 1] = [255, 255, 255]
    return location_map


def visualize_output(tensor, filename):
    location_map = tensor_to_image(tensor.detach().cpu().numpy())
    plt.imshow(location_map.numpy(), cmap='viridis')
    plt.title('Location Map Visualization')
    # img = tensor_to_image(tensor.detach().cpu().numpy())
    # plt.imshow(img)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
parser.add_argument('--batch-size', type=int, default=16, metavar='S')
# parser.add_argument('--data-folder', type=str, default="bedroom_6x6", metavar='S')
parser.add_argument('--num-workers', type=int, default=12, metavar='N')
parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
# parser.add_argument('--train-size', type=int, default=6000, metavar='N')
parser.add_argument('--save-dir', type=str, default="loc_test", metavar='S')
parser.add_argument('--ablation', type=str, default=None, metavar='S')
parser.add_argument('--lr', type=float, default=0.001, metavar='N')
parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
parser.add_argument('--centroid-weight', type=float, default=10, metavar="N")
args = parser.parse_args()

save_dir = args.save_dir
ensuredir(save_dir)

model = Model(num_classes=3, num_input_channels=3)

checkpoint_path = paths.pretrained_weights
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

counter = 0
test_loader = get_datasets_and_loaders(args, mode='inference')

for inputs, _ in tqdm(test_loader, desc='Processing'):
    inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(inputs)
    counter += 1
    output_image = outputs.squeeze(0)
    img_name = f"heatmap_vis_{counter}.png"
    visualize_output(output_image, img_name)
