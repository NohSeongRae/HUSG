import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
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
from data_utils import get_inference_loader
from torchvision.utils import save_image


def ensuredir(dirname):
    """Ensure a directory exists"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def tensor_to_image(tensor, temperature_pixel=0.8):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    probs = F.softmax(tensor, dim=1)

    # probs = probs.unsqueeze(0)

    print(probs.shape)

    image = probs.cpu()

    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(image[0], cmap="YlGnBu", annot=False, fmt=".2e")  # YlGnBu: 파란색 계열의 colormap
    plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    """

    """

    print(probs.shape)

    upsampled_tensor = F.interpolate(probs, size=(256, 256), mode='bilinear', align_corners=True).squeeze(0)

    location_map = upsampled_tensor
    location_map = location_map ** (1 / temperature_pixel)
    location_map = location_map / location_map.sum()

    print("location map: ", location_map.shape)

    print(location_map)
    """

    return probs


def visualize_output_channel0(tensor, filename):
    location_map = tensor_to_image(tensor).cpu()

    location_map_channel0 = location_map[0]

    # print("channel0", location_map_channel0.shape)


    # location_map = location_map.permute(1, 2, 0)
    plt.imshow(location_map_channel0, cmap='viridis')
    plt.title('Channel2 Visualization')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_output_channel1(tensor, filename):
    location_map = tensor_to_image(tensor).cpu()

    # location_map = location_map.permute(1, 2, 0)
    plt.imshow(location_map[1], cmap='viridis')
    plt.title('Channel1 Visualization')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__=="__main__":
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

    model = Model(num_classes=2, num_input_channels=2)

    weight = [args.centroid_weight for i in range(2)]  # 의문이 남음
    weight[0] = 1
    print(weight)

    weight = torch.from_numpy(np.asarray(weight)).float().cuda()
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    softmax = nn.Softmax()
    print(f'CUDA available: {torch.cuda.is_available()}')

    model.cuda()
    cross_entropy.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)
    checkpoint_path = paths.pretrained_weights
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    total_loss = 0.0
    counter = 0
    test_loader = get_inference_loader(args)


    with torch.no_grad():
        for inputs in tqdm(test_loader, desc='Processing'):
            inputs = inputs.cuda()
            outputs = model(inputs)

            counter += 1
            output_image = outputs.squeeze(0)

            if counter < 20:
                img_name_0 = f"./output/result/heatmap_vis_{counter}_0.png"
                visualize_output_channel0(output_image, img_name_0)

                img_name_1 = f"./output/result/heatmap_vis_{counter}_1.png"
                visualize_output_channel1(output_image, img_name_1)