import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch.distributed as dist

import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from boundary_transformer import get_pad_mask
from boundary_transformer import BoundaryTransformer
from boundary_dataloader import BoundaryDataset

import wandb

class Trainer:
    def __init__(self, batch_size, max_epoch, sos_idx, eos_idx, pad_idx, d_street, d_unit, d_model, n_layer, n_head,
                 n_building, n_boundary, dropout, use_checkpoint, checkpoint_epoch, use_tensorboard, val_epoch, save_epoch,
                 lr, n_street,
                 use_global_attn, use_street_attn, use_local_attn, local_rank, save_dir_path):
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
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
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
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.lr = lr
        self.use_global_attn = use_global_attn
        self.use_street_attn = use_street_attn
        self.use_local_attn = use_local_attn
        self.local_rank = local_rank
        self.save_dir_path = save_dir_path

        print('local_rank', self.local_rank)

        self.available_gpus = torch.cuda.device_count()

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

        # Only the first dataset initialization will load the full dataset from disk
        self.train_dataset = BoundaryDataset(sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx, n_street=n_street,
                                             n_boundary=n_boundary, d_street=d_street, data_type='train')
        self.train_sampler = torch.utils.data.DistributedSampler(dataset=self.train_dataset, num_replicas=self.available_gpus, rank=rank)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           sampler=self.train_sampler, num_workers=8)

        # Subsequent initializations will use the already loaded full dataset
        self.val_dataset = BoundaryDataset(sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx, n_street=n_street,
                                           n_boundary=n_boundary, d_street=d_street, data_type='val')
        self.val_sampler = torch.utils.data.DistributedSampler(dataset=self.val_dataset, num_replicas=self.available_gpus, rank=rank)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                         sampler=self.val_sampler, num_workers=8)

        # Initialize the Transformer model
        self.transformer = BoundaryTransformer(sos_idx=self.sos_idx, eos_idx=self.eos_idx, pad_idx=self.pad_idx,
                                               d_street=self.d_street, d_unit=self.d_unit, d_model=self.d_model,
                                               d_inner=self.d_model * 4, n_layer=self.n_layer, n_head=self.n_head,
                                               dropout=self.dropout,
                                               use_global_attn=use_global_attn,
                                               use_street_attn=use_street_attn,
                                               use_local_attn=use_local_attn).to(device=self.device)
        self.transformer = nn.parallel.DistributedDataParallel(self.transformer, device_ids=[local_rank], find_unused_parameters=True)

        # optimizer
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False)

        # scheduler
        data_len = len(self.train_dataloader)
        num_train_steps = int(data_len / batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_train_steps)

    def recon_loss(self, pred, trg, street_indices):
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed Recon loss.
        """
        loss = F.mse_loss(pred[:, 1:], trg[:, 1:], reduction='none')

        # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
        pad_mask = get_pad_mask(street_indices[:, 1:], pad_idx=0)
        mask = pad_mask.unsqueeze(-1).expand(-1, -1, 4)

        # mask 적용
        masked_loss = loss * mask.float()
        # 손실의 평균 반환
        return masked_loss.sum() / mask.sum()

    def smooth_loss(self, pred, street_indices):
        cur_token = pred[:, :-1, 2:]
        next_token = pred[:, 1:, :2]
        loss = F.mse_loss(cur_token, next_token, reduction='none')

        # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
        pad_mask = get_pad_mask(street_indices, pad_idx=0)
        mask = pad_mask.unsqueeze(-1).expand(-1, -1, 2)[:, :-1, :]

        masked_loss = loss * mask.float()

        return masked_loss.sum() / mask.sum()

    def train(self):
        """Training loop for the transformer model."""
        epoch_start = 0

        if self.use_checkpoint:
            checkpoint = torch.load("./models/" + self.save_dir_path + "/epoch_" + str(self.checkpoint_epoch) + ".pth")
            self.transformer.module.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']

        if self.use_tensorboard:
            self.writer = SummaryWriter()

        for epoch in range(epoch_start, self.max_epoch):
            total_recon_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
            total_smooth_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분

            # Iterate over batches
            for data in tqdm(self.train_dataloader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                src_unit_seq, src_street_seq, street_index_seq, gt_unit_seq = data
                src_unit_seq = src_unit_seq.to(device=self.device, dtype=torch.float32)
                src_street_seq = src_street_seq.to(device=self.device, dtype=torch.float32)
                street_index_seq = street_index_seq.to(device=self.device, dtype=torch.long)
                gt_unit_seq = gt_unit_seq.to(device=self.device, dtype=torch.float32)

                # Get the model's predictions
                output = self.transformer(src_unit_seq, src_street_seq, street_index_seq)

                # Compute the losses
                loss_recon = self.recon_loss(output, gt_unit_seq.detach(), street_index_seq.detach())
                loss_smooth = self.smooth_loss(output, street_index_seq.detach())
                loss_total = loss_recon + loss_smooth

                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 모든 GPU에서 손실 값을 합산 <-- 수정된 부분
                dist.all_reduce(loss_recon, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_smooth, op=dist.ReduceOp.SUM)
                total_recon_loss += loss_recon
                total_smooth_loss += loss_smooth

                # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
            if self.local_rank == 0:
                loss_recon_mean = total_recon_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_smooth_mean = total_smooth_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Recon: {loss_recon_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Smooth: {loss_smooth_mean:.4f}")

                if self.use_tensorboard:
                    wandb.log({"Train recon loss": loss_recon_mean, "epoch": epoch + 1})
                    wandb.log({"Train smooth loss": loss_smooth_mean, "epoch": epoch + 1})

            if (epoch + 1) % self.val_epoch == 0:
                self.transformer.module.eval()
                total_recon_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분
                total_smooth_loss = torch.Tensor([0.0]).to(self.device)  # <--- 추가된 부분

                with torch.no_grad():
                    # Iterate over batches
                    for data in tqdm(self.val_dataloader):
                        # Get the source and target sequences from the batch
                        src_unit_seq, src_street_seq, street_index_seq, gt_unit_seq = data
                        src_unit_seq = src_unit_seq.to(device=self.device, dtype=torch.float32)
                        src_street_seq = src_street_seq.to(device=self.device, dtype=torch.float32)
                        street_index_seq = street_index_seq.to(device=self.device, dtype=torch.long)
                        gt_unit_seq = gt_unit_seq.to(device=self.device, dtype=torch.float32)

                        output = self.transformer(src_unit_seq, src_street_seq, street_index_seq)

                        # Compute the losses
                        loss_recon = self.recon_loss(output, gt_unit_seq, street_index_seq)
                        loss_smooth = self.smooth_loss(output, street_index_seq)

                        # 모든 GPU에서 손실 값을 합산 <-- 수정된 부분
                        dist.all_reduce(loss_recon, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss_smooth, op=dist.ReduceOp.SUM)
                        total_recon_loss += loss_recon
                        total_smooth_loss += loss_smooth

                        # 첫 번째 GPU에서만 평균 손실을 계산하고 출력 <-- 수정된 부분
                    if self.local_rank == 0:
                        loss_recon_mean = total_recon_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_smooth_mean = total_smooth_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Recon: {loss_recon_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Smooth: {loss_smooth_mean:.4f}")

                        if self.use_tensorboard:
                            wandb.log({"Validation recon loss": loss_recon_mean, "epoch": epoch + 1})
                            wandb.log({"Validation smooth loss": loss_smooth_mean, "epoch": epoch + 1})

                self.transformer.module.train()

            if (epoch + 1) % self.save_epoch == 0:
                # 체크포인트 데이터 준비
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
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Maximum number of epochs for training.")
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
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--use_global_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_street_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_local_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="boundary_transformer", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")


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
    dist.init_process_group("nccl")

    if opt.local_rank == 0:
        wandb.login(key='5a8475b9b95df52a68ae430b3491fe9f67c327cd')
        wandb.init(project='boundary_transformer')
        # 실행 이름 설정
        wandb.run.name = 'fix smooth loss'
        wandb.run.save()
        wandb.config.update(opt)

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch, pad_idx=opt.pad_idx, n_street=opt.n_street,
                      sos_idx=opt.sos_idx, eos_idx=opt.eos_idx,
                      d_street=opt.d_street, d_unit=opt.d_unit, d_model=opt.d_model, n_layer=opt.n_layer, n_head=opt.n_head,
                      n_building=opt.n_building, n_boundary=opt.n_boundary, use_tensorboard=opt.use_tensorboard,
                      dropout=opt.dropout, use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      val_epoch=opt.val_epoch, save_epoch=opt.save_epoch, lr=opt.lr,
                      use_global_attn=opt.use_global_attn, use_street_attn=opt.use_street_attn, use_local_attn=opt.use_local_attn,
                      local_rank=opt.local_rank, save_dir_path=opt.save_dir_path)

    trainer.train()