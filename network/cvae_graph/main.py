import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MessagePassing
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch.distributed as dist
import matplotlib.pyplot as plt

import numpy as np
import random
from tqdm import tqdm

from model import GraphCVAE
from dataloader import GraphDataset

import wandb

class DistanceLossLayer(MessagePassing):
    def __init__(self):
        super(DistanceLossLayer, self).__init__(aggr='add')  # 'add' aggregation

    def forward(self, edge_index, pos, target_pos):
        # edge_index에 따라 거리를 계산하고 손실을 집계합니다.
        return self.propagate(edge_index, size=(pos.size(0), pos.size(0)), pos=pos, target_pos=target_pos)

    def message(self, pos_i, pos_j, target_pos_i, target_pos_j):
        # 노드 쌍 간의 목표 거리 차이를 계산합니다.
        target_distance_diff = torch.norm(target_pos_i - target_pos_j, p=2, dim=-1)
        # 노드 쌍 간의 실제 거리 차이를 계산합니다.
        actual_distance_diff = torch.norm(pos_i - pos_j, p=2, dim=-1)
        # 거리 차이의 제곱을 반환합니다.
        return (target_distance_diff - actual_distance_diff) ** 2

class Trainer:
    def __init__(self, batch_size, max_epoch, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 val_epoch, save_epoch, local_rank, save_dir_path, lr, T, d_feature, d_latent, n_head,
                 pos_weight, size_weight, theta_weight, kl_weight):
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
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.local_rank = local_rank
        self.save_dir_path = save_dir_path
        self.lr = lr
        self.T = T
        self.d_feature = d_feature
        self.d_latent = d_latent
        self.n_head = n_head
        self.pos_weight = pos_weight
        self.size_weight = size_weight
        self.theta_weight = theta_weight
        self.kl_weight = kl_weight

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
        self.cvae = GraphCVAE(T=T, feature_dim=d_feature, latent_dim=d_latent, n_head=n_head).to(device=self.device)
        self.cvae = nn.parallel.DistributedDataParallel(self.cvae, device_ids=[local_rank])

        # optimizer
        param_optimizer = list(self.cvae.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False,
                               no_deprecation_warning=True)

        # scheduler
        data_len = len(self.train_dataloader)
        num_train_steps = int(data_len / batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_train_steps)

    def recon_pos_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')
        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum() * self.pos_weight

    def recon_size_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')
        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum() * self.size_weight

    def recon_theta_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')
        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum() * self.theta_weight

    def kl_loss(self, mu, log_var):
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss * self.kl_weight

    def distance_loss(self, pred, trg, mask, edge_index):
        pred = pred * mask
        trg = trg * mask

        # edge_index에서 시작 노드와 끝 노드의 인덱스를 가져옵니다.
        start_nodes, end_nodes = edge_index

        # 실제 좌표와 목표 좌표를 사용하여 거리를 계산합니다.
        actual_distances = torch.norm(pred[start_nodes] - pred[end_nodes], dim=-1)
        target_distances = torch.norm(trg[start_nodes] - trg[end_nodes], dim=-1)

        # 거리 차이의 제곱을 계산합니다.
        loss = torch.sum((actual_distances - target_distances) ** 2)

        # 배치의 평균 손실을 반환합니다.
        return loss / mask.sum()

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
            if self.local_rank == 0:
                wandb.watch(self.cvae.module, log='all')  # <--- 추가된 부분

        for epoch in range(epoch_start, self.max_epoch):
            total_pos_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
            total_size_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
            total_theta_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
            total_kl_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the model's predictions
                data = data.to(device=self.device)
                output_pos, output_size, output_theta, mu, log_var = self.cvae(data)

                # Compute the losses
                loss_pos = self.recon_pos_loss(output_pos, data.building_feature.detach()[:, :2],
                                               data.building_mask.detach())
                loss_size = self.recon_size_loss(output_size, data.building_feature.detach()[:, 2:4],
                                                 data.building_mask.detach())
                loss_theta = self.recon_theta_loss(output_theta, data.building_feature.detach()[:, 4:],
                                                   data.building_mask.detach())
                loss_kl = self.kl_loss(mu, log_var)
                loss_distance = self.distance_loss(output_pos, data.building_feature.detach()[:, :2],
                                                   data.building_mask.detach(), data.edge_index.detach())
                print(loss_distance)
                loss_total = loss_pos + loss_size + loss_theta + loss_kl

                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 모든 GPU에서 손실 값을 합산 <-- 수정된 부분
                dist.all_reduce(loss_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_theta, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_kl, op=dist.ReduceOp.SUM)
                total_pos_loss += loss_pos
                total_size_loss += loss_size
                total_theta_loss += loss_theta
                total_kl_loss += loss_kl

                # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
            if self.local_rank == 0:
                loss_pos_mean = total_pos_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_size_mean = total_size_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_theta_mean = total_theta_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_kl_mean = total_kl_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Pos: {loss_pos_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Size: {loss_size_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Theta: {loss_theta_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss KL: {loss_kl_mean:.4f}")

                if self.use_tensorboard:
                    wandb.log({"Train pos loss": loss_pos_mean}, step=epoch + 1)
                    wandb.log({"Train size loss": loss_size_mean}, step=epoch + 1)
                    wandb.log({"Train theta loss": loss_theta_mean}, step=epoch + 1)
                    wandb.log({"Train kl loss": loss_kl_mean}, step=epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.cvae.module.eval()
                total_pos_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
                total_size_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
                total_theta_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
                total_kl_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분

                with torch.no_grad():
                    # Iterate over batches
                    for data in tqdm(self.val_dataloader):
                        # Get the source and target sequences from the batch
                        data = data.to(device=self.device)
                        output_pos, output_size, output_theta, mu, log_var = self.cvae(data)

                        # Compute the losses using the generated sequence
                        loss_pos = self.recon_pos_loss(output_pos, data.building_feature[:, :2], data.building_mask)
                        loss_size = self.recon_size_loss(output_size, data.building_feature[:, 2:4], data.building_mask)
                        loss_theta = self.recon_theta_loss(output_theta, data.building_feature[:, 4:],
                                                           data.building_mask)
                        loss_kl = self.kl_loss(mu, log_var)

                        # 모든 GPU에서 손실 값을 합산 <-- 수정된 부분
                        dist.all_reduce(loss_pos, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_size, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_theta, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_kl, op=dist.ReduceOp.SUM)
                        total_pos_loss += loss_pos
                        total_size_loss += loss_size
                        total_theta_loss += loss_theta
                        total_kl_loss += loss_kl

                        # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
                    if self.local_rank == 0:
                        loss_pos_mean = total_pos_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_size_mean = total_size_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_theta_mean = total_theta_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_kl_mean = total_kl_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Pos: {loss_pos_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Size: {loss_size_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Theta: {loss_theta_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss KL: {loss_kl_mean:.4f}")

                        if self.use_tensorboard:
                            wandb.log({"Validation pos loss": loss_pos_mean}, step=epoch + 1)
                            wandb.log({"Validation size loss": loss_size_mean}, step=epoch + 1)
                            wandb.log({"Validation theta loss": loss_theta_mean}, step=epoch + 1)
                            wandb.log({"Validation kl loss": loss_kl_mean}, step=epoch + 1)

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
                    torch.save(checkpoint, os.path.join(save_path, "epoch_" + str(epoch + 1) + ".pth"))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a cvae with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Maximum number of epochs for training.")
    parser.add_argument("--T", type=int, default=3, help="Dimension of the model.")
    parser.add_argument("--d_feature", type=int, default=256, help="Dimension of the model.")
    parser.add_argument("--d_latent", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_head", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="cvae_graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--size_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--theta_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--kl_weight", type=float, default=0.5, help="save dir path")

    opt = parser.parse_args()

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    if opt.local_rank == 0:
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

    if opt.local_rank == 0:
        wandb.login(key='5a8475b9b95df52a68ae430b3491fe9f67c327cd')
        wandb.init(project='cvae_graph')
        # 실행 이름 설정
        wandb.run.name = 'cvae init'
        wandb.run.save()
        wandb.config.update(opt)

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch,
                      d_feature=opt.d_feature, d_latent=opt.d_latent, n_head=opt.n_head, T=opt.T,
                      use_tensorboard=opt.use_tensorboard,
                      use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      local_rank=opt.local_rank, save_dir_path=opt.save_dir_path, lr=opt.lr,
                      pos_weight=opt.pos_weight, size_weight=opt.size_weight, theta_weight=opt.theta_weight,
                      kl_weight=opt.kl_weight)

    trainer.train()
