import sys
import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm

from network.graph_based_mid_layer.model import VQVAE
from network.graph_based_mid_layer.dataloader import VQVAEDataset

class Trainer:
    def __init__(self, batch_size, max_epoch, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 train_ratio, val_ratio, test_ratio, val_epoch, save_epoch,
                 weight_decay, scheduler_step, scheduler_gamma,
                 d_in, d_model, d_out, n_embed, n_res_layer, commitment_cost):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of image_based_vqvae layers.
        - n_head (int): Number of multi-head attentions.
        """

        # Initialize trainer parameters
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_embed = n_embed
        self.n_res_layer = n_res_layer
        self.commitment_cost = commitment_cost

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize the dataset and dataloader
        self.train_dataset = VQVAEDataset(train_ratio=self.train_ratio,
                                          val_ratio=self.val_ratio,
                                          test_ratio=self.test_ratio,
                                          data_type='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = VQVAEDataset(train_ratio=self.train_ratio,
                                        val_ratio=self.val_ratio,
                                        test_ratio=self.test_ratio,
                                        data_type='val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Transformer model
        self.vqvae = VQVAE(d_in=d_in, d_model=d_model, d_out=d_out, n_embed=n_embed,
                           n_res_layer=n_res_layer, commitment_cost=commitment_cost).to(device=self.device)

        # Set the optimizer for the training process
        self.optimizer = torch.optim.Adam(self.vqvae.parameters(),
                                          lr=5e-4,
                                          betas=(0.9, 0.98),
                                          weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.scheduler_step], gamma=self.scheduler_gamma)

    def recon_loss(self, pred, trg):
        loss = F.mse_loss(pred, trg)
        return loss

    def train(self):
        """Training loop for the image_based_vqvae model."""
        epoch_start = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/vqvae_epoch_" + str(self.checkpoint_epoch) + ".pt")
            self.vqvae.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()

        for epoch in tqdm(range(epoch_start, self.max_epoch)):
            loss_recon_mean = 0
            loss_vq_mean = 0

            # Iterate over batches
            for data in self.train_dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                input_data, target_data = data
                input_data = input_data.to(device=self.device, dtype=torch.float32)
                target_data = target_data.to(device=self.device, dtype=torch.float32)

                # Get the model's predictions
                output, loss_vq = self.vqvae(input_data)

                # Compute the losses
                loss_recon = self.recon_loss(output, target_data)
                loss_total = loss_recon + loss_vq

                # Accumulate the losses for reporting
                loss_recon_mean += loss_recon.detach().item()
                loss_vq_mean += loss_vq.detach().item()

                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()

            self.scheduler.step()

            # Print the average losses for the current epoch
            loss_recon_mean /= len(self.train_dataloader)
            loss_vq_mean /= len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Recon: {loss_recon_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss VQ: {loss_vq_mean:.4f}")

            if self.use_tensorboard:
                self.writer.add_scalar("Train/loss-recon", loss_recon_mean, epoch + 1)
                self.writer.add_scalar("Train/loss-vq", loss_vq_mean, epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.vqvae.eval()
                loss_recon_mean = 0
                loss_vq_mean = 0

                with torch.no_grad():
                    # Iterate over batches
                    for data in self.val_dataloader:
                        # Get the source and target sequences from the batch
                        input_data, target_data = data
                        input_data = input_data.to(device=self.device, dtype=torch.float32)
                        target_data = target_data.to(device=self.device, dtype=torch.long)

                        # Get the model's predictions
                        output, loss_vq = self.vqvae(input_data)

                        # Compute the losses
                        loss_recon = self.recon_loss(output, target_data)

                        # Accumulate the losses for reporting
                        loss_recon_mean += loss_recon.detach().item()
                        loss_vq_mean += loss_vq.detach().item()

                    # Print the average losses for the current epoch
                    loss_recon_mean /= len(self.val_dataloader)
                    loss_vq_mean /= len(self.train_dataloader)
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Recon: {loss_recon_mean:.4f}")
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss VQ: {loss_vq_mean:.4f}")

                    if self.use_tensorboard:
                        self.writer.add_scalar("Val/loss-recon", loss_recon_mean, epoch + 1)
                        self.writer.add_scalar("Val/loss-vq", loss_vq_mean, epoch + 1)

            if (epoch + 1) % self.save_epoch == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.vqvae.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, "./models/vqvae_epoch_" + str(epoch + 1) + ".pt")

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a image_based_vqvae with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=False, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=50, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=50, help="Use checkpoint index.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Use checkpoint index.")
    parser.add_argument("--scheduler_step", type=int, default=200, help="Use checkpoint index.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--d_in", type=int, default=3, help="Use checkpoint index.")
    parser.add_argument("--d_model", type=int, default=256, help="Use checkpoint index.")
    parser.add_argument("--d_out", type=int, default=3, help="Use checkpoint index.")
    parser.add_argument("--n_embed", type=int, default=100, help="Use checkpoint index.")
    parser.add_argument("--n_res_layer", type=int, default=2, help="Use checkpoint index.")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Use checkpoint index.")


    opt = parser.parse_args()

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch, use_tensorboard=opt.use_tensorboard,
                      use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      weight_decay=opt.weight_decay, scheduler_step=opt.scheduler_step, scheduler_gamma=opt.scheduler_gamma,
                      d_in=opt.d_in, d_model=opt.d_model, d_out=opt.d_out, n_embed=opt.n_embed,
                      n_res_layer=opt.n_res_layer, commitment_cost=opt.commitment_cost)
    trainer.train()