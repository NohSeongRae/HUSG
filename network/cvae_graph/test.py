import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import numpy as np
import random
from tqdm import tqdm

from model import GraphCVAE
from dataloader import GraphDataset
from visualization import plot


def recon_pos_loss(pred, trg, mask):
    recon_loss = F.mse_loss(pred, trg, reduction='none')
    recon_loss = recon_loss * mask
    return recon_loss.sum() / mask.sum()


def recon_size_loss(pred, trg, mask):
    recon_loss = F.mse_loss(pred, trg, reduction='none')
    recon_loss = recon_loss * mask
    return recon_loss.sum() / mask.sum()


def recon_theta_loss(pred, trg, mask):
    recon_loss = F.mse_loss(pred, trg, reduction='none')
    recon_loss = recon_loss * mask
    return recon_loss.sum() / mask.sum()


def kl_loss(mu, log_var):
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_loss

def test(d_feature, d_latent, n_head, T, checkpoint_epoch, save_dir_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Subsequent initializations will use the already loaded full dataset
    test_dataset = GraphDataset(data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Initialize the Transformer model
    cvae = GraphCVAE(T=T, feature_dim=d_feature, latent_dim=d_latent, n_head=n_head).to(device=device)

    checkpoint = torch.load("./models/" + save_dir_path + "/epoch_" + str(checkpoint_epoch) + ".pth")
    cvae.load_state_dict(checkpoint['model_state_dict'])

    cvae.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            # Get the source and target sequences from the batch
            data = data.to(device=device)
            output_pos, output_size, output_theta = cvae.test(data)

            # # Compute the losses using the generated sequence
            # loss_pos = recon_pos_loss(output_pos, data.building_feature[:, :2], data.building_mask)
            # loss_size = recon_size_loss(output_size, data.building_feature[:, 2:4], data.building_mask)
            # loss_theta = recon_theta_loss(output_theta, data.building_feature[:, 4:], data.building_mask)
            # loss_kl = kl_loss(mu, log_var)
            #
            # print(f"Epoch {idx + 1}/{len(test_dataloader)} - Validation Loss Pos: {loss_pos:.4f}")
            # print(f"Epoch {idx + 1}/{len(test_dataloader)} - Validation Loss Size: {loss_size:.4f}")
            # print(f"Epoch {idx + 1}/{len(test_dataloader)} - Validation Loss Theta: {loss_theta:.4f}")
            # print(f"Epoch {idx + 1}/{len(test_dataloader)} - Validation Loss KL: {loss_kl:.4f}")

            plot(data.detach().cpu().numpy(),
                 output_pos.detach().cpu().numpy(),
                 output_size.detach().cpu().numpy(),
                 output_theta.detach().cpu().numpy()[0])

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--sos_idx", type=int, default=2, help="Padding index for sequences.")
    parser.add_argument("--eos_idx", type=int, default=3, help="Padding index for sequences.")
    parser.add_argument("--pad_idx", type=int, default=4, help="Padding index for sequences.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--d_street", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--d_unit", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_building", type=int, default=120, help="binary classification for building existence.")
    parser.add_argument("--n_boundary", type=int, default=250, help="Number of boundary or token.")
    parser.add_argument("--n_street", type=int, default=60, help="Number of boundary or token.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--use_global_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_street_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_local_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="default_path", help="save dir path")

    opt = parser.parse_args()

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    test(d_feature=opt.d_feature, d_latent=opt.d_latent, n_head=opt.n_head, T=opt.T,
         checkpoint_epoch=opt.checkpoint_epoch, save_dir_path=opt.save_dir_path)