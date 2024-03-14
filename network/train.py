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

import numpy as np
import random

from model import GraphTransformer
from dataloader import GraphDataset

import wandb

class Trainer:
    def __init__(self, batch_size, max_epoch, use_checkpoint, checkpoint_epoch, use_tensorboard, dropout,
                 val_epoch, save_epoch, local_rank, save_dir_path, lr, d_model, n_layer, n_head, weight_decay):
        """
        Initializes the Trainer class.

        Parameters:
        - batch_size (int): Batch size for training and validation.
        - max_epoch (int): Maximum number of epochs for training.
        - use_checkpoint (bool): Whether to use a pre-trained model checkpoint.
        - checkpoint_epoch (int): The epoch of the checkpoint to use.
        - use_tensorboard (bool): Whether to log data to TensorBoard.
        - dropout (float): Dropout rate used in the transformer model.
        - val_epoch (int): Frequency of validation in terms of epochs.
        - save_epoch (int): Frequency of saving the model in terms of epochs.
        - local_rank (int): The rank of the current process in the distributed training setup.
        - save_dir_path (str): Path to save the model checkpoints.
        - lr (float): Learning rate for the optimizer.
        - d_model (int): Dimension of the model (embeddings).
        - n_layer (int): Number of layers in the transformer model.
        - n_head (int): Number of attention heads in the transformer model.
        - weight_decay (float): Weight decay (L2 penalty) for the optimizer.

        Initializes training and validation datasets, dataloaders, model, and optimizer. Sets up distributed training if applicable.
        """

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
        Calculates the cross-entropy loss with padding mask applied.

        Parameters:
        - pred (Tensor): Predictions from the model.
        - trg (Tensor): Target values.
        - pad_mask (Tensor): Padding mask to ignore certain positions in the input.

        Returns:
        - Tensor: The average loss computed over all non-masked elements.
        """

        loss = F.binary_cross_entropy(pred, trg, reduction='none')

        masked_loss = loss * pad_mask.float() + loss * trg
        return masked_loss.sum() / pad_mask.float().sum()

    def edge_sum_loss(self, pred, trg, pad_mask):
        """
        Calculates the mean squared error loss for the sum of edges, considering the padding.

        Parameters:
        - pred (Tensor): Predictions from the model.
        - trg (Tensor): Target values.
        - pad_mask (Tensor): Padding mask to exclude padded areas from the loss calculation.

        Returns:
        - Tensor: The average loss computed over all non-masked elements, specifically for the sum of edges.
        """

        pred = pred.reshape(-1, pred.shape[-1])
        trg = trg.reshape(-1, trg.shape[-1])
        pad_mask = pad_mask.reshape(-1, pad_mask.shape[-1])

        pred = pred * pad_mask
        loss = F.mse_loss(torch.sum(pred, dim=-1), torch.sum(trg, dim=-1), reduction='none')

        masked_loss = loss * pad_mask[:, 0].float()
        return masked_loss.sum() / pad_mask.float().sum()

    def correct_data(self, pred, trg, pad_mask):
        """
        Computes the number of correctly predicted data points, considering the padding.

        Parameters:
        - pred (Tensor): Predictions from the model, with a threshold applied to determine binary outcomes.
        - trg (Tensor): Target values.
        - pad_mask (Tensor): Padding mask to consider only non-padded elements in the accuracy calculation.

        Returns:
        - tuple: A tuple containing the number of correct predictions and the number of non-masked elements, used to compute accuracy.
        """

        predictions = (pred >= 0.5).float()

        correct_predictions = (predictions == trg).float() * pad_mask

        masked_elements_count = pad_mask.sum().item()

        return correct_predictions.sum().item(), masked_elements_count

    def train(self):
        """
        Executes the training loop over the dataset for a specified number of epochs.
        Validates the model at specified intervals, saves model checkpoints, and logs to TensorBoard/WandB.

        Returns:
        None
        """

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
            total_exist_loss = torch.Tensor([0.0]).to(self.device)
            total_count_loss = torch.Tensor([0.0]).to(self.device)
            total_loss = torch.Tensor([0.0]).to(self.device)
            total_correct = torch.Tensor([0.0]).to(self.device)
            total_problem = torch.Tensor([0.0]).to(self.device)

            for data in self.train_dataloader:
                self.optimizer.zero_grad()

                building_adj_matrix_padded = data['building_adj_matrix_padded'].to(device=self.device)
                boundary_adj_matrix_padded = data['boundary_adj_matrix_padded'].to(device=self.device)
                bb_adj_matrix_padded = data['bb_adj_matrix_padded'].to(device=self.device)
                boundary_pos_padded = data['boundary_pos_padded'].to(device=self.device)
                building_pad_mask = data['building_pad_mask'].to(device=self.device)
                boundary_pad_mask = data['boundary_pad_mask'].to(device=self.device)
                bb_pad_mask = data['bb_pad_mask'].to(device=self.device)

                output = self.transformer(building_adj_matrix_padded, boundary_adj_matrix_padded,
                                          building_pad_mask, boundary_pad_mask, boundary_pos_padded)

                exist_loss = self.cross_entropy_loss(output, bb_adj_matrix_padded.detach(), bb_pad_mask.detach())
                count_loss = self.edge_sum_loss(output, bb_adj_matrix_padded.detach(), bb_pad_mask.detach()) + \
                             self.edge_sum_loss(output.permute(0, 2, 1), bb_adj_matrix_padded.permute(0, 2, 1).detach(), bb_pad_mask.permute(0, 2, 1).detach())
                loss = exist_loss + count_loss
                correct, problem = self.correct_data(output.detach(), bb_adj_matrix_padded.detach(), bb_pad_mask.detach())

                loss.backward()
                self.optimizer.step()

                dist.all_reduce(exist_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(problem), op=dist.ReduceOp.SUM)


                total_exist_loss += exist_loss
                total_count_loss += count_loss
                total_loss += loss
                total_correct += correct
                total_problem += problem
                # if self.local_rank == 0:
                #     print(loss_pos, loss_size, loss_kl)

            if self.local_rank == 0:
                exist_loss_mean = total_exist_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                count_loss_mean = total_count_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                loss_mean = total_loss.item() / (len(self.train_dataloader) * dist.get_world_size())
                correct_mean = total_correct.item() / total_problem.item()

                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Total : {loss_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Exist : {exist_loss_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Count : {count_loss_mean:.4f}")
                print(f"Epoch {epoch + 1}/{self.max_epoch} - Accuracy : {correct_mean:.4f}")

                if self.use_tensorboard:
                    wandb.log({"Train loss total": loss_mean}, step=epoch + 1)
                    wandb.log({"Train loss exist": exist_loss_mean}, step=epoch + 1)
                    wandb.log({"Train loss count": count_loss_mean}, step=epoch + 1)
                    wandb.log({"Train accuracy": correct_mean}, step=epoch + 1)

            if (epoch + 1) % self.val_epoch == 0:
                self.transformer.module.eval()
                total_exist_loss = torch.Tensor([0.0]).to(self.device)
                total_count_loss = torch.Tensor([0.0]).to(self.device)
                total_loss = torch.Tensor([0.0]).to(self.device)
                total_correct = torch.Tensor([0.0]).to(self.device)
                total_problem = torch.Tensor([0.0]).to(self.device)

                with torch.no_grad():
                    for data in self.val_dataloader:
                        building_adj_matrix_padded = data['building_adj_matrix_padded'].to(device=self.device)
                        boundary_adj_matrix_padded = data['boundary_adj_matrix_padded'].to(device=self.device)
                        bb_adj_matrix_padded = data['bb_adj_matrix_padded'].to(device=self.device)
                        boundary_pos_padded = data['boundary_pos_padded'].to(device=self.device)
                        building_pad_mask = data['building_pad_mask'].to(device=self.device)
                        boundary_pad_mask = data['boundary_pad_mask'].to(device=self.device)
                        bb_pad_mask = data['bb_pad_mask'].to(device=self.device)

                        output = self.transformer(building_adj_matrix_padded, boundary_adj_matrix_padded,
                                                  building_pad_mask, boundary_pad_mask, boundary_pos_padded)

                        exist_loss = self.cross_entropy_loss(output.detach(), bb_adj_matrix_padded.detach(), bb_pad_mask.detach())
                        count_loss = self.edge_sum_loss(output.detach(), bb_adj_matrix_padded.detach(), bb_pad_mask.detach()) + \
                                     self.edge_sum_loss(output.permute(0, 2, 1), bb_adj_matrix_padded.permute(0, 2, 1).detach(), bb_pad_mask.permute(0, 2, 1).detach())
                        loss = exist_loss + count_loss
                        correct, problem = self.correct_data(output.detach(), bb_adj_matrix_padded.detach(), bb_pad_mask.detach())

                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(exist_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(count_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(torch.tensor(correct), op=dist.ReduceOp.SUM)
                        dist.all_reduce(torch.tensor(problem), op=dist.ReduceOp.SUM)

                        total_exist_loss += exist_loss
                        total_count_loss += count_loss
                        total_loss += loss
                        total_correct += correct
                        total_problem += problem

                    if self.local_rank == 0:
                        loss_exist_mean = total_exist_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_count_mean = total_count_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        loss_mean = total_loss.item() / (len(self.val_dataloader) * dist.get_world_size())
                        correct_mean = total_correct.item() / total_problem.item()

                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Total : {loss_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Exist : {loss_exist_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Loss Count : {loss_count_mean:.4f}")
                        print(f"Epoch {epoch + 1}/{self.max_epoch} - Validation Accuracy : {correct_mean:.4f}")

                        if self.use_tensorboard:
                            wandb.log({"Validation loss total": loss_mean}, step=epoch + 1)
                            wandb.log({"Validation loss exist": loss_exist_mean}, step=epoch + 1)
                            wandb.log({"Validation loss count": loss_count_mean}, step=epoch + 1)
                            wandb.log({"Validation accuracy": correct_mean}, step=epoch + 1)

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

    parser.add_argument("--batch_size", type=int, default=32, help="The batch size used during training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="The maximum number of epochs to train for.")
    parser.add_argument("--d_model", type=int, default=512, help="The dimensionality of the model's embeddings.")
    parser.add_argument("--n_layer", type=int, default=6, help="The number of layers in the transformer model.")
    parser.add_argument("--n_head", type=int, default=8, help="The number of attention heads in the transformer model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="The dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="A seed for random number generation to ensure reproducibility.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Flag to enable logging to TensorBoard..")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Flag to enable loading the model from a checkpoint.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="The epoch number of the checkpoint to load for training continuation.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Frequency (in epochs) with which to perform validation.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Frequency (in epochs) with which to save model checkpoints.")
    parser.add_argument("--local-rank", type=int, help="The local rank of the process for distributed training. Required for multi-GPU setups.")
    parser.add_argument("--save_dir_path", type=str, default="transformer_graph", help="Directory path where model checkpoints will be saved")
    parser.add_argument("--lr", type=float, default=3e-5, help="The learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="The weight decay (L2 penalty) used by the optimizer")

    opt = parser.parse_args()

    if opt.local_rank == 0:
        wandb.login(key='key')
        wandb.init(project='project', config=vars(opt))

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