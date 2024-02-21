import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import matplotlib.pyplot as plt

import numpy as np
import random
from tqdm import tqdm

from model import GraphCVAE
from dataloader import GraphDataset

import wandb

class Trainer:
    def __init__(self, batch_size, max_epoch, use_checkpoint, checkpoint_epoch, use_tensorboard,
                 val_epoch, save_epoch, local_rank, save_dir_path, lr, T, d_feature, d_latent, n_head,
                 pos_weight, size_weight, theta_weight, kl_weight, distance_weight,
                 condition_type, convlayer, weight_decay):
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
        self.weight_decay = weight_decay
        self.T = T
        self.d_feature = d_feature
        self.d_latent = d_latent
        self.n_head = n_head
        self.pos_weight = pos_weight
        self.size_weight = size_weight
        self.theta_weight = theta_weight
        self.kl_weight = kl_weight
        self.distance_weight = distance_weight
        self.condition_type = condition_type
        self.convlayer = convlayer

        print('local_rank', self.local_rank)

        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        self.train_dataset = GraphDataset(data_type='train',
                                          condition_type=condition_type)
        self.train_sampler = DistributedSampler(dataset=self.train_dataset, rank=rank, shuffle=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           sampler=self.train_sampler, num_workers=8, pin_memory=True)

        self.val_dataset = GraphDataset(data_type='val',
                                        condition_type=condition_type)
        self.val_sampler = DistributedSampler(dataset=self.val_dataset, rank=rank, shuffle=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                         sampler=self.val_sampler, num_workers=8, pin_memory=True)

        self.cvae = GraphCVAE(T=T, feature_dim=d_feature, latent_dim=d_latent, n_head=n_head,
                              condition_type=condition_type,
                              convlayer=convlayer).to(device=self.device)
        self.cvae = nn.parallel.DistributedDataParallel(self.cvae, device_ids=[local_rank])

        base_batch_size = 16
        new_batch_size = self.batch_size
        self.lr = self.lr * (new_batch_size / base_batch_size)

        self.optimizer = torch.optim.Adam(self.cvae.module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay,
                                          betas=(0.9, 0.98))

    def recon_pos_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')

        if mask is None:
            return recon_loss.mean()

        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum()

    def recon_size_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')

        if mask is None:
            return recon_loss.mean()

        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum()

    def recon_theta_loss(self, pred, trg, mask):
        recon_loss = F.mse_loss(pred, trg, reduction='none')

        if mask is None:
            return recon_loss.mean()

        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum()

    def recon_exist_loss(self, pred, trg, mask):
        # pred와 trg 간의 binary cross entropy loss 계산
        recon_loss = F.binary_cross_entropy(pred.float(), trg.float(), reduction='none')

        # mask가 제공되지 않은 경우
        if mask is None:
            return recon_loss.mean()

        # mask 적용
        recon_loss = recon_loss * mask
        return recon_loss.sum() / mask.sum()

    def kl_loss(self, mu, log_var):
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss

    def distance_loss(self, pred, trg, mask, edge_index):
        if mask is not None:
            mask = ((mask[edge_index[0]] == 1) & (mask[edge_index[1]] == 1)).squeeze(-1)
            selected_edge_index = edge_index[:, mask]

            start_nodes, end_nodes = selected_edge_index
        else:
            start_nodes, end_nodes = edge_index

        actual_distances = torch.norm(pred[start_nodes] - pred[end_nodes], dim=-1)
        target_distances = torch.norm(trg[start_nodes] - trg[end_nodes], dim=-1)

        loss = torch.sum(torch.abs(actual_distances - target_distances))
        return loss / len(start_nodes)

    def train(self):
        epoch_start = 0
        min_loss = 999
        early_stop_count = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/cvae/epoch_" + str(self.checkpoint_epoch) + ".pth")
            self.cvae.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()
            if self.local_rank == 0:
                wandb.watch(self.cvae.module, log='all')

        for epoch in range(epoch_start, self.max_epoch):
            total_pos_loss = torch.Tensor([0.0]).to(self.device)
            total_size_loss = torch.Tensor([0.0]).to(self.device)
            total_theta_loss = torch.Tensor([0.0]).to(self.device)
            total_exist_loss = torch.Tensor([0.0]).to(self.device)
            total_kl_loss = torch.Tensor([0.0]).to(self.device)
            total_distance_loss = torch.Tensor([0.0]).to(self.device)

            for data in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()

                data = data.to(device=self.device)
                output_pos, output_size, output_theta, output_exist, mu, log_var = self.cvae(data)

                mask = data.building_mask.detach()
                gt_feature = data.node_features
                gt_exist = data.exist_features

                loss_pos = self.recon_pos_loss(output_pos, gt_feature.detach()[:, :2], mask)
                loss_size = self.recon_size_loss(output_size, gt_feature.detach()[:, 2:4], mask)
                loss_theta = self.recon_theta_loss(output_theta, gt_feature.detach()[:, 4:], mask)
                loss_exist = self.recon_exist_loss(output_exist, gt_exist.detach(), mask)
                loss_kl = self.kl_loss(mu, log_var)
                loss_distance = self.distance_loss(output_pos, gt_feature.detach()[:, :2],
                                                   mask, data.edge_index.detach())# 각 손실 출력

                loss_total = loss_pos * self.pos_weight + loss_size * self.size_weight + \
                             loss_theta * self.theta_weight + loss_kl * self.kl_weight + \
                             loss_distance * self.distance_weight + loss_exist

                loss_total.backward()
                self.optimizer.step()

                dist.all_reduce(loss_pos, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_theta, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_exist, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_kl, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_distance, op=dist.ReduceOp.SUM)
                total_pos_loss += loss_pos
                total_size_loss += loss_size
                total_theta_loss += loss_theta
                total_exist_loss += loss_exist
                total_kl_loss += loss_kl
                total_distance_loss += loss_distance

            if self.local_rank == 0:
                loss_pos_mean = total_pos_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_size_mean = total_size_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_theta_mean = total_theta_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_exist_mean = total_exist_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_kl_mean = total_kl_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_distance_mean = total_distance_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Pos: {loss_pos_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Size: {loss_size_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Theta: {loss_theta_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Exist: {loss_exist_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss KL: {loss_kl_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Distance: {loss_distance_mean:.4f}")

                if self.use_tensorboard:
                    wandb.log({"Train pos loss": loss_pos_mean}, step=epoch + 1)
                    wandb.log({"Train size loss": loss_size_mean}, step=epoch + 1)
                    wandb.log({"Train theta loss": loss_theta_mean}, step=epoch + 1)
                    wandb.log({"Train exist loss": loss_exist_mean}, step=epoch + 1)
                    wandb.log({"Train kl loss": loss_kl_mean}, step=epoch + 1)
                    wandb.log({"Train distance loss": loss_distance_mean}, step=epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.cvae.module.eval()
                total_pos_loss = torch.Tensor([0.0]).to(self.device)
                total_size_loss = torch.Tensor([0.0]).to(self.device)
                total_theta_loss = torch.Tensor([0.0]).to(self.device)
                total_exist_loss = torch.Tensor([0.0]).to(self.device)
                total_kl_loss = torch.Tensor([0.0]).to(self.device)
                total_distance_loss = torch.Tensor([0.0]).to(self.device)

                with torch.no_grad():
                    for data in tqdm(self.val_dataloader):
                        data = data.to(device=self.device)
                        output_pos, output_size, output_theta, output_exist, mu, log_var = self.cvae(data)

                        mask = data.building_mask
                        gt_feature = data.node_features
                        gt_exist = data.exist_features

                        loss_pos = self.recon_pos_loss(output_pos, gt_feature[:, :2], mask)
                        loss_size = self.recon_size_loss(output_size, gt_feature[:, 2:4], mask)
                        loss_theta = self.recon_theta_loss(output_theta, gt_feature[:, 4:], mask)
                        loss_exist = self.recon_exist_loss(output_exist, gt_exist, mask)
                        loss_kl = self.kl_loss(mu, log_var)
                        loss_distance = self.distance_loss(output_pos, gt_feature[:, :2],
                                                           mask, data.edge_index)

                        dist.all_reduce(loss_pos, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_size, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_theta, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_exist, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_kl, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_distance, op=dist.ReduceOp.SUM)
                        total_pos_loss += loss_pos
                        total_size_loss += loss_size
                        total_theta_loss += loss_theta
                        total_exist_loss += loss_exist
                        total_kl_loss += loss_kl
                        total_distance_loss += loss_distance

                    if self.local_rank == 0:
                        loss_pos_mean = total_pos_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_size_mean = total_size_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_theta_mean = total_theta_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_exist_mean = total_exist_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_kl_mean = total_kl_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_distance_mean = total_distance_loss.item() / (
                                len(self.val_dataloader) * dist.get_world_size())
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Pos: {loss_pos_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Size: {loss_size_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Theta: {loss_theta_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Exist: {loss_exist_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss KL: {loss_kl_mean:.4f}")
                        print(
                            f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Distance: {loss_distance_mean:.4f}")

                        if self.use_tensorboard:
                            wandb.log({"Validation pos loss": loss_pos_mean}, step=epoch + 1)
                            wandb.log({"Validation size loss": loss_size_mean}, step=epoch + 1)
                            wandb.log({"Validation theta loss": loss_theta_mean}, step=epoch + 1)
                            wandb.log({"Validation exist loss": loss_exist_mean}, step=epoch + 1)
                            wandb.log({"Validation kl loss": loss_kl_mean}, step=epoch + 1)
                            wandb.log({"Validation distance loss": loss_distance_mean}, step=epoch + 1)

                            loss_total = loss_pos_mean + loss_size_mean + loss_theta_mean + loss_kl_mean + loss_distance_mean + loss_exist_mean
                            wandb.log({"Validation total loss": loss_total}, step=epoch + 1)

                            if min_loss > loss_total:
                                min_loss = loss_total
                                checkpoint = {
                                    'epoch': epoch,
                                    'model_state_dict': self.cvae.module.state_dict(),
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

                self.cvae.module.train()

            if (epoch + 1) % self.save_epoch == 0:
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
    parser = argparse.ArgumentParser(description="Initialize a cvae with user-defined hyperparameters.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=500, help="Maximum number of epochs for training.")
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
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="save dir path")
    parser.add_argument("--pos_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--size_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--theta_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--kl_weight", type=float, default=0.5, help="save dir path")
    parser.add_argument("--distance_weight", type=float, default=4.0, help="save dir path")
    parser.add_argument("--condition_type", type=str, default='image_resnet34', help="save dir path")
    parser.add_argument("--convlayer", type=str, default='gat', help="save dir path")

    opt = parser.parse_args()

    if opt.local_rank == 0:
        wandb.login(key='5a8475b9b95df52a68ae430b3491fe9f67c327cd')
        wandb.init(project='cvae_graph', config=vars(opt))

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
                      d_feature=opt.d_feature, d_latent=opt.d_latent, n_head=opt.n_head, T=opt.T,
                      use_tensorboard=opt.use_tensorboard,
                      use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch,
                      local_rank=opt.local_rank, save_dir_path=opt.save_dir_path, lr=opt.lr,
                      pos_weight=opt.pos_weight, size_weight=opt.size_weight, theta_weight=opt.theta_weight,
                      kl_weight=opt.kl_weight, distance_weight=opt.distance_weight, condition_type=opt.condition_type,
                      convlayer=opt.convlayer, weight_decay=opt.weight_decay)

    trainer.train()

    if opt.local_rank == 0:
        wandb.finish()