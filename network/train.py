import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import matplotlib.pyplot as plt

import numpy as np
import random
from tqdm import tqdm

from model import GraphTransformer
from dataloader import GraphDataset

import wandb

class Trainer:
    def __init__(self, batch_size, max_epoch, use_checkpoint, checkpoint_epoch, use_tensorboard, dropout,
                 val_epoch, save_epoch, local_rank, save_dir_path, lr, d_model, n_layer, n_head, weight_decay):
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
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.local_rank = local_rank
        self.save_dir_path = save_dir_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_head = n_head
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout = dropout

        print('local_rank', self.local_rank)

        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        self.train_dataset = GraphDataset(data_type='train')
        self.train_sampler = DistributedSampler(dataset=self.train_dataset, rank=rank, shuffle=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           sampler=self.train_sampler, num_workers=8, pin_memory=True)

        self.val_dataset = GraphDataset(data_type='val')
        self.val_sampler = DistributedSampler(dataset=self.val_dataset, rank=rank, shuffle=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                         sampler=self.val_sampler, num_workers=8, pin_memory=True)

        self.transformer = GraphTransformer(d_model, d_model*4, n_layer, n_head, dropout).to(device=self.device)
        self.transformer = nn.parallel.DistributedDataParallel(self.transformer, device_ids=[local_rank])

        self.optimizer = torch.optim.Adam(self.transformer.module.parameters(),
                                          lr=self.lr,
                                          betas=(0.9, 0.98))

    def cross_entropy_loss(self, pred, trg, pad_mask):
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed BCE loss.
        """
        loss = F.binary_cross_entropy(pred, trg, reduction='none')

        # mask 적용
        masked_loss = loss * pad_mask.float()
        # 손실의 평균 반환
        return masked_loss.sum() / pad_mask.float().sum()

    def train(self):
        epoch_start = 0
        min_loss = 99999999999
        early_stop_count = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/transformer/epoch_" + str(self.checkpoint_epoch) + ".pth")
            self.transformer.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()
            # if self.local_rank == 0:
            #     wandb.watch(self.transformer.module, log='all')

        for epoch in range(epoch_start, self.max_epoch):
            total_loss = torch.Tensor([0.0]).to(self.device)
            total_correct = torch.Tensor([0.0]).to(self.device)

            for data in self.train_dataloader:
                self.optimizer.zero_grad()

                building_adj_matrix_padded = torch.tensor(data['building_adj_matrix_padded'], dtype=torch.float32).to(device=self.device)
                boundary_adj_matrix_padded = torch.tensor(data['boundary_adj_matrix_padded'], dtype=torch.float32).to(device=self.device)
                bb_adj_matrix_padded = torch.tensor(data['bb_adj_matrix_padded'], dtype=torch.float32).to(device=self.device)
                building_pad_mask = torch.tensor(data['building_pad_mask'], dtype=torch.bool).to(device=self.device)
                boundary_pad_mask = torch.tensor(data['boundary_pad_mask'], dtype=torch.bool).to(device=self.device)
                bb_pad_mask = torch.tensor(data['bb_pad_mask'], dtype=torch.bool).to(device=self.device)

                output = self.transformer(building_adj_matrix_padded, boundary_adj_matrix_padded,
                                          building_pad_mask, boundary_pad_mask)

                loss = self.cross_entropy_loss(output, bb_adj_matrix_padded.detach(), bb_pad_mask)

                loss_total.backward()
                self.optimizer.step()

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)

                total_loss += loss
                # if self.local_rank == 0:
                #     print(loss_pos, loss_size, loss_kl)

            if self.local_rank == 0:
                loss_mean = total_loss.item() / (len(self.train_dataloader) * dist.get_world_size())

                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss : {loss_mean:.4f}")

                if self.use_tensorboard:
                    wandb.log({"Train loss": loss_mean}, step=epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.transformer.module.eval()
                total_loss = torch.Tensor([0.0]).to(self.device)
                total_correct = torch.Tensor([0.0]).to(self.device)

                with torch.no_grad():
                    for data in self.val_dataloader:
                        building_adj_matrix_padded = torch.tensor(data['building_adj_matrix_padded'], dtype=torch.float32).to(device=self.device)
                        boundary_adj_matrix_padded = torch.tensor(data['boundary_adj_matrix_padded'], dtype=torch.float32).to(device=self.device)
                        bb_adj_matrix_padded = torch.tensor(data['bb_adj_matrix_padded'], dtype=torch.float32).to(device=self.device)
                        building_pad_mask = torch.tensor(data['building_pad_mask'], dtype=torch.bool).to(device=self.device)
                        boundary_pad_mask = torch.tensor(data['boundary_pad_mask'], dtype=torch.bool).to(device=self.device)
                        bb_pad_mask = torch.tensor(data['bb_pad_mask'], dtype=torch.bool).to(device=self.device)

                        output = self.transformer(building_adj_matrix_padded, boundary_adj_matrix_padded,
                                                  building_pad_mask, boundary_pad_mask)

                        loss = self.cross_entropy_loss(output.detach(), bb_adj_matrix_padded.detach(), bb_pad_mask)

                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

                        total_loss += loss

                    if self.local_rank == 0:
                        loss_mean = total_loss.item() / (len(self.val_dataloader) * dist.get_world_size())

                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss: {loss_mean:.4f}")

                        if self.use_tensorboard:
                            wandb.log({"Validation loss": loss_mean}, step=epoch + 1)

                            loss_total = loss_mean
                            if min_loss > loss_total:
                                min_loss = loss_total
                                checkpoint = {
                                    'epoch': epoch,
                                    'model_state_dict': self.transformer.module.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                }

                                save_path = os.path.join("./models", self.save_dir_path)
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                torch.save(checkpoint, os.path.join(save_path, "epoch_best.pth"))

                                early_stop_count = 0
                            else:
                                early_stop_count += 1
                                if early_stop_count >= self.max_epoch / 10:
                                    break

                self.transformer.module.train()

            if (epoch + 1) % self.save_epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.transformer.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

                if self.local_rank == 0:
                    save_path = os.path.join("./models", self.save_dir_path)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(checkpoint, os.path.join(save_path, "epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="transformer_graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="save dir path")

    opt = parser.parse_args()

    if opt.local_rank == 0:
        wandb.login(key='5a8475b9b95df52a68ae430b3491fe9f67c327cd')
        wandb.init(project='transformer_graph', config=vars(opt))

        for key, value in wandb.config.items():
            setattr(opt, key, value)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_dir_path = f"{opt.save_dir_path}_{current_time}"

    if opt.local_rank == 0:
        save_path = os.path.join("./models", opt.save_dir_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        config_file_path = os.path.join(save_path, "config.txt")
        with open(config_file_path, "w") as f:
            for arg in vars(opt):
                f.write(f"{arg}: {getattr(opt, arg)}\n")

    if opt.local_rank == 0:
        for arg in vars(opt):
            print(f"{arg}: {getattr(opt, arg)}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    rank = opt.local_rank
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        if torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') == "cuda:0":
            dist.init_process_group("gloo")

        else:
            dist.init_process_group('nccl')

    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      d_model=opt.d_model, n_layer=opt.n_layer, n_head=opt.n_head, dropout=opt.dropout,
                      use_tensorboard=opt.use_tensorboard,
                      use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      local_rank=opt.local_rank, save_dir_path=opt.save_dir_path, lr=opt.lr, weight_decay=opt.weight_decay)

    trainer.train()

    if opt.local_rank == 0:
        wandb.finish()