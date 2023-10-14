import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm

from network.blockplanner.dataloader import BlockplannerDataset
from network.blockplanner.model import BlockPlanner

class Trainer():
    def __init__(self, batch_size, max_epoch, d_model, n_iter, n_building, n_semantic,
                 use_checkpoint, checkpoint_epoch, use_tensorboard, w_r, w_x, w_s, w_g, w_v):
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.d_model = d_model
        self.n_iter = n_iter
        self.n_building = n_building
        self.n_semantic = n_semantic
        self.use_checkpoint = use_checkpoint
        self.checkpoint_epoch = checkpoint_epoch
        self.use_tensorboard = use_tensorboard

        self.w_r = w_r
        self.w_x = w_x
        self.w_s = w_s
        self.w_g = w_g
        self.w_v = w_v

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize the dataset and dataloader
        self.dataset = BlockplannerDataset(n_building=n_building, n_semantic=n_semantic)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Blockplanner model
        self.model = BlockPlanner(n_building=self.n_building,
                                  n_semantic=self.n_semantic,
                                  d_model=self.d_model,
                                  n_iter=self.n_iter).to(device=self.device)

        # Set the optimizer for the training process
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=5e-4,
                                          betas=(0.9, 0.98))
    def calc_reconstruction_loss(self, pred_geometry, pred_aspect_ratio, real_geometry, real_aspect_ratio):
        loss_geometry = F.l1_loss(pred_geometry, real_geometry, reduction='mean')
        loss_aspect_ratio = F.l1_loss(pred_aspect_ratio, real_aspect_ratio, reduction='mean')

        loss = self.w_r * (loss_geometry + loss_aspect_ratio) / 2
        return loss

    def calc_existence_loss(self, pred_lot_prob, real_lot_prob, pred_edge_prob, real_edge_prob, pred_adj_matrix, real_adj_matrix):
        loss_lot = F.binary_cross_entropy(pred_lot_prob, real_lot_prob)
        loss_edge = F.binary_cross_entropy(pred_edge_prob, real_edge_prob)

        mask_weight = 1.5 * (real_adj_matrix == 1).float() + (real_adj_matrix == 0).float()
        loss_adj_matrix = mask_weight * F.l1_loss(pred_adj_matrix, real_adj_matrix)
        loss_adj_matrix = loss_adj_matrix.mean()

        loss = self.w_x * (loss_lot + loss_edge + loss_adj_matrix)
        return loss

    def calc_landuse_loss(self, pred_land_use, real_land_use):
        loss = self.w_g * F.cross_entropy(pred_land_use, real_land_use)
        return loss

    def calc_geometric_validation(self, pred_geometry, real_geometry):
        loss = self.w_g * F.l1_loss(pred_geometry, real_geometry, reduction='mean')
        return loss

    def calc_variational_regularization_loss(self, mu, log_var):
        loss = self.w_v * (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))

        return loss
    def train(self):
        for epoch in range(0, self.max_epoch):
            recon_loss_mean = 0
            exist_loss_mean = 0
            landuse_loss_mean = 0
            geometric_loss_mean = 0
            regularization_loss_mean = 0

            for data in self.dataloader:
                self.optimizer.zero_grad()

                data = data.to(device=self.device)
                outputs = self.model(data)

                recon_loss = self.calc_reconstruction_loss(pred_geometry=outputs['lot_geometry'],
                                                           real_geometry=data.geometry,
                                                           pred_aspect_ratio=outputs['aspect_ratio'],
                                                           real_aspect_ratio=data.aspect_ratio)
                exist_loss = self.calc_existence_loss(pred_lot_prob=outputs['lot_exists_prob'],
                                                      real_lot_prob=data.node_exists_prob,
                                                      pred_edge_prob=outputs['adj_matrix'],
                                                      real_edge_prob=data.adj_matrix,
                                                      pred_adj_matrix=outputs['adj_matrix'],
                                                      real_adj_matrix=data.adj_matrix)
                landuse_loss = self.calc_landuse_loss(pred_land_use=outputs['land_use_attribute'],
                                                      real_land_use=data.semantic)
                geometric_loss = self.calc_geometric_validation(pred_geometry=outputs['lot_geometry'],
                                                                real_geometry=data.geometry)
                regularization_loss = self.calc_variational_regularization_loss(mu=outputs['mu'],
                                                                                log_var=outputs['log_var'])

                recon_loss_mean += recon_loss.detach().item()
                exist_loss_mean += exist_loss.detach().item()
                landuse_loss_mean += landuse_loss.detach().item()
                geometric_loss_mean += geometric_loss.detach().item()
                regularization_loss_mean += regularization_loss.detach().item()

                loss_total = recon_loss + exist_loss + landuse_loss + geometric_loss + regularization_loss
                loss_total.backward()

                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Recon Mean: {recon_loss_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Exist Mean: {exist_loss_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss LandUse Mean: {landuse_loss_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss Geometric Mean: {geometric_loss_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss KL Mean: {regularization_loss_mean:.4f}")


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=400, help="Maximum number of epochs for training.")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model.")
    parser.add_argument("--n_iter", type=int, default=3, help="iteration number for message passing.")
    parser.add_argument("--n_building", type=int, default=30, help="binary classification for building existence.")
    parser.add_argument("--n_semantic", type=int, default=11, help="binary classification for building existence.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=False, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--w_r", type=float, default=1, help="Use checkpoint index.")
    parser.add_argument("--w_x", type=float, default=0.5, help="Use checkpoint index.")
    parser.add_argument("--w_s", type=float, default=0.2, help="Use checkpoint index.")
    parser.add_argument("--w_g", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--w_v", type=float, default=0.2, help="Use checkpoint index.")

    opt = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch,  d_model=opt.d_model, n_iter=opt.n_iter,
                      n_building=opt.n_building, use_tensorboard=opt.use_tensorboard, n_semantic=opt.n_semantic,
                      use_checkpoint=opt.use_checkpoint, checkpoint_epoch=opt.checkpoint_epoch,
                      w_r=opt.w_r, w_x=opt.w_x, w_s=opt.w_s, w_g=opt.w_g, w_v=opt.w_v)
    trainer.train()