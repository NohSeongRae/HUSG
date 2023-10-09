import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm

from network.transformer.transformer import Transformer
from network.transformer.dataloader import BoundaryDataset

class Trainer:
    def __init__(self, batch_size, max_epoch, pad_idx, d_model, n_layer, n_head,
                 n_building, n_boundary, dropout):
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
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_building = n_building
        self.n_boundary = n_boundary
        self.dropout = dropout

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize the dataset and dataloader
        self.dataset = BoundaryDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Transformer model
        self.transformer = Transformer(n_building=self.n_building, n_boundary=self.n_boundary, pad_idx=self.pad_idx, d_model=self.d_model,
                                       d_inner=self.d_model * 4, n_layer=self.n_layer, n_head=self.n_head,
                                       d_k=self.d_model//self.n_head, d_v=self.d_model//self.n_head,
                                       dropout=self.dropout).to(device=self.device)

        # Set the optimizer for the training process
        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=5e-4,
                                          betas=(0.9, 0.98))

    def cross_entropy_loss(self, pred, trg):
        """
        Compute the binary cross-entropy loss between predictions and targets.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Computed BCE loss.
        """

        probs = F.sigmoid(pred)
        loss = - (trg * torch.log(probs) + (1 - trg) * torch.log(1 - probs))
        loss = loss.mean()
        return loss

    def center_position_loss(self, pred, trg, pos_table, top_p=0.9):
        """
        Compute the center position loss between predicted and actual positions.

        Args:
        - pred (torch.Tensor): Model predictions.
        - trg (torch.Tensor): Ground truth probabilities.
        - pos_table (torch.Tensor): Position table for mapping.

        Returns:
        - torch.Tensor: Computed center position loss.
        """

        logits = torch.log(F.softmax(pred, dim=-1))

        batch_size, sequence_length, vocab_size = logits.shape

        # Flatten logits to 2D [batch_size * sequence_length, vocab_size]
        logits_2d = logits.view(-1, vocab_size)

        # Apply top-p sampling for each sequence position and each candidate
        sorted_logits, sorted_indices = torch.sort(logits_2d, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs > top_p
        shift_one_right = torch.cat([torch.zeros_like(remove_mask[..., :1]), remove_mask[..., :-1]], dim=-1)
        remove_mask = torch.logical_or(remove_mask, shift_one_right)

        # Using a loop to handle the scatter operation
        for i in range(logits_2d.shape[0]):
            indices_to_remove = sorted_indices[i, remove_mask[i]]
            logits_2d[i].scatter_(0, indices_to_remove, -float('Inf'))

        # Sample from the flattened logits
        num_candidates = self.n_building // 2
        sampled_indices_2d = torch.multinomial(F.softmax(logits_2d, dim=-1), num_candidates)

        # Reshape to [batch_size, sequence_length, num_candidates]
        sampled_indices = sampled_indices_2d.view(batch_size, sequence_length, num_candidates)
        self.pad_idx = 5
        pad_mask = (sampled_indices == self.pad_idx).float()

        # Use cumsum to mark everything after the first pad_idx as pad
        cumulative_pad_mask = pad_mask.cumsum(dim=-1)
        pad_mask = (cumulative_pad_mask == 0)

        sampled_indices = sampled_indices * pad_mask
        pred_one_hot_indices = F.one_hot(sampled_indices, num_classes=vocab_size)

        threshold = 0.9
        gt_one_hot_indices = torch.rand_like(pred_one_hot_indices.to(dtype=torch.float32)) > threshold
        b1_one_hot_indices = gt_one_hot_indices & ~pred_one_hot_indices
        b2_one_hot_indices = pred_one_hot_indices & ~gt_one_hot_indices

        b1_probs = b1_one_hot_indices * F.softmax(pred, dim=-1).unsqueeze(2).expand(-1, -1, num_candidates, -1)
        b1_probs = b1_probs.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, 2)

        b2_probs = b2_one_hot_indices * F.softmax(pred, dim=-1).unsqueeze(2).expand(-1, -1, num_candidates, -1)
        b2_probs = b2_probs.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, 2)

        b1_pos_out = torch.matmul(b1_one_hot_indices.to(dtype=torch.float32), pos_table)
        b2_pos_out = torch.matmul(b2_one_hot_indices.to(dtype=torch.float32), pos_table)

        b1_pos_out = b1_pos_out * b1_probs
        b2_pos_out = b2_pos_out * (1-b2_probs)
        bc_pos = torch.ones_like(b1_pos_out)




    def train(self):
        """Training loop for the transformer model."""

        for epoch in tqdm(range(0, self.max_epoch)):
            loss_ce_mean = 0
            loss_cp_mean = 0

            # Iterate over batches
            for data in self.dataloader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Get the source and target sequences from the batch
                src_unit_seq, src_street_seq, trg_seq = data
                src_unit_seq = src_unit_seq.to(device=self.device, dtype=torch.float32)
                src_street_seq = src_street_seq.to(device=self.device, dtype=torch.float32)
                trg_seq = trg_seq.to(device=self.device, dtype=torch.long)

                # Get the model's predictions
                output = self.transformer(src_unit_seq, src_street_seq, trg_seq)

                # Compute the losses
                loss_ce = self.cross_entropy_loss(output, output)
                loss_cp = self.center_position_loss(output, output, pos_table=torch.rand((10, 2)).to(self.device, dtype=torch.float32))
                loss_total = loss_ce + loss_cp

                # Accumulate the losses for reporting
                loss_ce_mean += loss_ce.detach().item()
                loss_cp_mean += loss_cp.detach().item()

                # Backpropagation and optimization step
                loss_total.backward()
                self.optimizer.step()

            # Print the average losses for the current epoch
            loss_ce_mean /= len(self.dataloader)
            loss_cp_mean /= len(self.dataloader)
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss CE: {loss_ce_mean:.4f}")
            print(f"Epoch {epoch + 1}/{self.max_epoch} - Loss CP: {loss_cp_mean:.4f}")

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=10, help="Maximum number of epochs for training.")
    parser.add_argument("--pad_idx", type=int, default=0, help="Padding index for sequences.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_building", type=int, default=10, help="Number of building blocks.")
    parser.add_argument("--n_boundary", type=int, default=20, help="Number of boundary or token.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Create a Trainer instance and start the training process
    trainer = Trainer(batch_size=opt.batch_size, max_epoch=opt.max_epoch, pad_idx=opt.pad_idx,
                      d_model=opt.d_model, n_layer=opt.n_layer, n_head=opt.n_head,
                      n_building=opt.n_building, n_boundary=opt.n_boundary,
                      dropout=opt.dropout)
    trainer.train()
