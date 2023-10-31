import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch.distributed as dist

import numpy as np
import random
from tqdm import tqdm

from model import GraphCVAE
from dataloader import GraphDataset

class Trainer:
    def __init__(self, batch_size, max_epoch, d_model, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 val_epoch, save_epoch, local_rank, save_dir_path):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of cvae layers.
        - n_head (int): Number of multi-head attentions.
        """

        # Initialize trainer parameters
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.local_rank = local_rank
        self.save_dir_path = save_dir_path

        print('local_rank', self.local_rank)

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        # Only the first dataset initialization will load the full dataset from disk
        self.train_dataset = GraphDataset(data_type='train')
        self.train_sampler = DistributedSampler(dataset=self.train_dataset, rank=rank)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           sampler=self.train_sampler, num_workers=8, pin_memory=True)

        # Subsequent initializations will use the already loaded full dataset
        self.val_dataset = GraphDataset(data_type='val')
        self.val_sampler = DistributedSampler(dataset=self.val_dataset, rank=rank)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                         sampler=self.val_sampler, num_workers=8, pin_memory=True)

        # Initialize the Transformer model
        self.cvae = GraphCVAE(T=3, feature_dim=256, latent_dim=256, n_head=8).to(device=self.device)
        self.cvae = nn.parallel.DistributedDataParallel(self.cvae, device_ids=[local_rank], find_unused_parameters=True)

        # optimizer
        param_optimizer = list(self.cvae.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, correct_bias=False)

        # scheduler
        data_len = len(self.train_dataloader)
        num_train_steps = int(data_len / batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_train_steps)

    def recon_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')
        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum()

    def kl_loss(self, mu, log_var):
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss

    def train(self):
        """Training loop for the cvae model."""
        epoch_start = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/cvae/epoch_" + str(self.checkpoint_epoch) + ".pth")
            self.cvae.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()

        for epoch in range(epoch_start, self.max_epoch):
            loss_recon_mean = 0
            loss_kl_mean = 0

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the model's predictions
                data = data.to(device=self.device)
                output, mu, log_var = self.cvae(data)

                # Compute the losses
                loss_recon = self.recon_loss(output, data.building_feature.detach(), data.building_mask.detach())
                loss_kl = self.kl_loss(mu, log_var)
                loss_total = loss_recon + loss_kl

                # Accumulate the losses for reporting
                loss_recon_mean += loss_recon.detach().item()
                loss_kl_mean += loss_kl.detach().item()
                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Print the average losses for the current epoch
            loss_recon_mean /= len(self.train_dataloader)
            loss_kl_mean /= len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Recon: {loss_recon_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss KL: {loss_kl_mean:.4f}")

            if self.use_tensorboard:
                self.writer.add_scalar("Train/loss-recon", loss_recon_mean, epoch + 1)
                self.writer.add_scalar("Train/loss-kl", loss_kl_mean, epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.cvae.module.eval()
                loss_recon_mean = 0
                loss_kl_mean = 0

                with torch.no_grad():
                    # Iterate over batches
                    for data in tqdm(self.val_dataloader):
                        # Get the source and target sequences from the batch
                        data = data.to(device=self.device)
                        output, mu, log_var = self.cvae(data)

                        # Compute the losses using the generated sequence
                        loss_recon = self.recon_loss(output, data.building_feature.detach(), data.building_mask.detach())
                        loss_kl = self.kl_loss(mu, log_var)
                        loss_recon_mean += loss_recon.detach().item()
                        loss_kl_mean += loss_kl.detach().item()

                    # Print the average losses for the current epoch
                    loss_recon_mean /= len(self.val_dataloader)
                    loss_kl_mean /= len(self.val_dataloader)
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Recon: {loss_recon_mean:.4f}")
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss KL: {loss_kl_mean:.4f}")

                    if self.use_tensorboard:
                        self.writer.add_scalar("Val/loss-recon", loss_recon_mean, epoch + 1)
                        self.writer.add_scalar("Val/loss-kl", loss_kl_mean, epoch + 1)

                self.cvae.module.train()

            if (epoch + 1) % self.save_epoch == 0:
                # 체크포인트 데이터 준비
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.cvae.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

                if self.local_rank == 0:
                    save_path = os.path.join("./models", self.save_dir_path)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(checkpoint, os.path.join(save_path, "cvae/epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a cvae with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Maximum number of epochs for training.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=False, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
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

    # ddp
    rank = opt.local_rank
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        if torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') == "cuda:0":
            dist.init_process_group("gloo")

        else:
            dist.init_process_group("nccl")

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      d_model=opt.d_model, use_tensorboard=opt.use_tensorboard,
                      use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      local_rank=opt.local_rank, save_dir_path=opt.save_dir_path)

    trainer.train()