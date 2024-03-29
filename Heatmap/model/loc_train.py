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
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from heatmap_models import Model
import paths
from data_utils import get_datasets_and_loaders
import random


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(f'runs/loc_experiment/{current_time}')

def save_input_images(img_tensor, save_dir):
    """
    Save images from a 2-channel tensor to a directory
    :param img_tensor: tensor of images, shape [B, C, H, W]
    :param save_dir: directory to save the images
    """
    ensuredir(save_dir)  # Ensure the directory exists
    for i in range(img_tensor.shape[0]):
        for j in range(img_tensor.shape[1]):  # Loop over channels
            filename = os.path.join(save_dir, f"image_{i}_channel_{j}.png")
            # Use unsqueeze to add the channel dimension back to the image
            save_image(img_tensor[i, j].unsqueeze(0), filename)


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


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
    parser.add_argument('--batch-size', type=int, default=16, metavar='S')
    parser.add_argument('--num-workers', type=int, default=12, metavar='N')
    parser.add_argument('--last-epoch', type=int, default=-1, metavar='N')
    parser.add_argument('--save-dir', type=str, default="loc_test", metavar='S')
    parser.add_argument('--ablation', type=str, default=None, metavar='S')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N')
    parser.add_argument('--eps', type=float, default=1e-6, metavar='N')
    parser.add_argument('--centroid-weight', type=float, default=10, metavar="N")
    parser.add_argument('--train-sample', type=bool, default=False, metavar="N")
    parser.add_argument('--use-total-shuffle', type=bool, default=False, metavar="N")
    parser.add_argument('--use-Kfold', type=bool, default=False, metavar="N")
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    save_dir = args.save_dir
    ensuredir(save_dir)

    num_categories = 1  # building이 있을 수 있는 곳, 있을 수 없는 곳
    num_input_channels = 2  # 3, if text encoding encluded
    logfile = open(f"{save_dir}/log_location.txt", 'w')


    def LOG(msg):
        print(msg)
        logfile.write(msg + '\n')
        logfile.flush()


    LOG('Building model...')

    weight = [args.centroid_weight for i in range(num_categories + 1)]
    weight[0] = 1
    print(weight)
    device = torch.device(f"cuda:{args.cuda_device}")
    print(device)
    weight = torch.from_numpy(np.asarray(weight)).float().to(device)
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    # softmax = nn.Softmax()
    print(f'CUDA available: {torch.cuda.is_available()}')
    LOG('Converting to CUDA')

    cross_entropy.to(device)
    if args.use_Kfold:
        loaders = get_datasets_and_loaders(args, 5)  # This will return a list of (train_loader, val_loader) pairs.
        fold_results = []

        MAX_EPOCHS = args.epoch
    else:
        train_loader, val_loader = get_datasets_and_loaders(args)
        MAX_EPOCHS=args.epoch
        model = Model(num_classes=num_categories + 1, num_input_channels=num_input_channels)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-6)

    def train(args, epoch, foldnum=5):
        model.train()
        if args.use_Kfold:
            epoch = epoch + foldnum * MAX_EPOCHS
        else:
            epoch = epoch
        print(f'Training Epoch: {epoch}')
        global current_epoch
        train_loader_progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(train_loader_progress):
            data, target = data.to(device), target.to(device)

            if epoch == 0:
                if batch_idx == 0:
                    save_input_images(data, "./check")

            target = target.squeeze(1)

            target = target.long()

            optimizer.zero_grad()
            output = model(data)
            loss = cross_entropy(output, target)
            train_loader_progress.set_postfix(loss=loss.item())

            loss.backward()
            optimizer.step()

            writer.add_scalar('loss ', loss.item(), epoch)

            if epoch % 10 == 0:
                # summary.add_scalar('loss ', loss, epoch)
                torch.save(model.state_dict(), f"{save_dir}/location_{epoch}.pt")
                torch.save(optimizer.state_dict(), f"{save_dir}/location_optim_backup.pt")


    def validate(epoch):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(val_loader):
                data, target = data.to(device), target.to(device)
                target = target.squeeze(1)
                target = target.long()
                output = model(data)
                loss = cross_entropy(output, target)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        writer.add_scalar('val_loss', avg_loss, epoch)
        return avg_loss


    if args.use_Kfold:
        for fold_num, (train_loader, val_loader) in enumerate(loaders):
            LOG(f'========================= Starting Fold {fold_num + 1} of {len(loaders)} =========================')
            for epoch in range(MAX_EPOCHS):
                LOG(f'===================================== Epoch {epoch} =====================================')
                train(args,epoch, fold_num)  # Train the model using the train_loader
                val_loss = validate(fold_num)  # Validate the model using the val_loader
                LOG(f'Validation Loss for Fold {fold_num + 1}: {val_loss}')
            fold_results.append(val_loss)  # only read last val_loss
        avg_validation_loss = sum(fold_results) / len(fold_results)
        LOG(f'Average Validation Loss over {len(loaders)} folds: {avg_validation_loss}')
    else:
        for epoch in range(MAX_EPOCHS):
            LOG(f'===================================== Epoch {epoch} =====================================')
            train(args, epoch)
            val_loss = validate(epoch)
            LOG(f'Validation Loss: {val_loss}')