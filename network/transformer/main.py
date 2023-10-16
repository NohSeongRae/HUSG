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

from transformer import get_pad_mask
from transformer import Transformer
from dataloader import BoundaryDataset
import loss
class Trainer:
    def __init__(self, batch_size, max_epoch, pad_idx, d_street, d_unit, d_model, n_layer, n_head,
                 n_building, n_boundary, dropout, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 train_ratio, val_ratio, test_ratio, data_type, val_epoch, save_epoch,
                 weight_decay, scheduler_step, scheduler_gamma):
        """
        Initialize the trainer with the specified parameters.

        Args:
        - batch_size (int): Size of each training batch.
        - max_epoch (int): Maximum number of training epochs.
        - pad_idx (int): Padding index for sequences.
        - d_model (int): Dimension of the model.
        - n_layer (int): Number of transformer layers.
        - n_head (int): Number of multi-head attentions.
        """

        # Initialize trainer parameters
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.pad_idx = pad_idx
        self.eos_idx = 2
        self.d_model = d_model
        self.d_street = d_street
        self.d_unit = d_unit
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_building = n_building
        self.n_boundary = n_boundary
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.data_type = data_type
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize the dataset and dataloader
        self.train_dataset = BoundaryDataset(train_ratio=self.train_ratio,
                                             val_ratio=self.val_ratio,
                                             test_ratio=self.test_ratio,
                                             data_type='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = BoundaryDataset(train_ratio=self.train_ratio,
                                           val_ratio=self.val_ratio,
                                           test_ratio=self.test_ratio,
                                           data_type='val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Transformer model
        self.transformer = Transformer(n_building=self.n_building, n_boundary=self.n_boundary, pad_idx=self.pad_idx,
                                       d_street=self.d_street, d_unit=self.d_unit, d_model=self.d_model,
                                       d_inner=self.d_model * 4, n_layer=self.n_layer, n_head=self.n_head,
                                       d_k=self.d_model//self.n_head, d_v=self.d_model//self.n_head,
                                       dropout=self.dropout, eos_idx=self.eos_idx).to(device=self.device)

        # Set the optimizer for the training process
        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=5e-4,
                                          betas=(0.9, 0.98),
                                          weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.scheduler_step], gamma=self.scheduler_gamma)


    def cross_entropy_loss(self, pred, trg):
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed BCE loss.
        """
        loss = F.binary_cross_entropy(torch.sigmoid(pred[:, :-1]), trg[:, 1:], reduction='none')

        # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
        mask = get_pad_mask(trg[:, 1:], pad_idx=self.eos_idx).float()

        # mask 적용
        masked_loss = loss * mask
        # 손실의 평균 반환
        return masked_loss.mean()

    def train(self):
        """Training loop for the transformer model."""
        epoch_start = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/transformer_epoch_" + str(self.checkpoint_epoch) + ".pt")
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()

        for epoch in tqdm(range(epoch_start, self.max_epoch)):
            loss_mean = 0

            # Iterate over batches
            for data in self.train_dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                src_unit_seq, src_street_seq, trg_building_seq, trg_street_seq = data
                gt_building_seq = trg_building_seq.to(device=self.device, dtype=torch.float32)
                src_unit_seq = src_unit_seq.to(device=self.device, dtype=torch.float32)
                src_street_seq = src_street_seq.to(device=self.device, dtype=torch.float32)
                trg_building_seq = trg_building_seq.to(device=self.device, dtype=torch.long)
                trg_street_seq = trg_street_seq.to(device=self.device, dtype=torch.long)

                # Get the model's predictions
                output = self.transformer(src_unit_seq, src_street_seq,
                                          trg_building_seq, trg_street_seq)

                # Compute the losses
                loss = self.cross_entropy_loss(output, gt_building_seq.detach())
                loss_total = loss

                # Accumulate the losses for reporting
                loss_mean += loss.detach().item()
                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()

            self.scheduler.step()

            # Print the average losses for the current epoch
            loss_mean /= len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss CE: {loss_mean:.4f}")

            if self.use_tensorboard:
                self.writer.add_scalar("Train/loss-obj", loss_mean, epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.transformer.eval()
                loss_mean = 0

                with torch.no_grad():
                    # Iterate over batches
                    for data in self.val_dataloader:
                        # Get the source and target sequences from the batch
                        src_unit_seq, src_street_seq, trg_building_seq, trg_street_seq = data
                        gt_building_seq = trg_building_seq.to(device=self.device, dtype=torch.float32)
                        src_unit_seq = src_unit_seq.to(device=self.device, dtype=torch.float32)
                        src_street_seq = src_street_seq.to(device=self.device, dtype=torch.float32)
                        trg_building_seq = trg_building_seq.to(device=self.device, dtype=torch.long)
                        trg_street_seq = trg_street_seq.to(device=self.device, dtype=torch.long)

                        # Get the model's predictions
                        output = self.transformer(src_unit_seq, src_street_seq,
                                                  trg_building_seq, trg_street_seq)

                        # Compute the losses
                        loss = self.cross_entropy_loss(output, gt_building_seq.detach())

                        # Accumulate the losses for reporting
                        loss_mean += loss.detach().item()

                    # Print the average losses for the current epoch
                    loss_mean /= len(self.val_dataloader)
                    print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss CE: {loss_mean:.4f}")

                    if self.use_tensorboard:
                        self.writer.add_scalar("Val/loss-obj", loss_mean, epoch + 1)

            if (epoch + 1) % self.save_epoch == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.transformer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, "./models/transformer_epoch_" + str(epoch) + ".pt")

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--pad_idx", type=int, default=0, help="Padding index for sequences.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--d_street", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--d_unit", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_building", type=int, default=1, help="binary classification for building existence.")
    parser.add_argument("--n_boundary", type=int, default=200, help="Number of boundary or token.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--data_type", type=str, default='train', help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=50, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=50, help="Use checkpoint index.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Use checkpoint index.")
    parser.add_argument("--scheduler_step", type=int, default=200, help="Use checkpoint index.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Use checkpoint index.")


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
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch, pad_idx=opt.pad_idx,
                      d_street=opt.d_street, d_unit=opt.d_unit, d_model=opt.d_model, n_layer=opt.n_layer, n_head=opt.n_head,
                      n_building=opt.n_building, n_boundary=opt.n_boundary, use_tensorboard=opt.use_tensorboard,
                      dropout=opt.dropout, use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
                      data_type=opt.data_type, val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      weight_decay=opt.weight_decay, scheduler_step=opt.scheduler_step, scheduler_gamma=opt.scheduler_gamma)
    trainer.train()
