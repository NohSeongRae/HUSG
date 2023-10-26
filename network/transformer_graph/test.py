import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import numpy as np
import random
from tqdm import tqdm

from model import get_pad_mask, get_subsequent_mask, get_clipped_adj_matrix
from model import GraphTransformer
from dataloader import GraphDataset
from visualization import plot

def make_upper_follow_lower_torch_padded(matrix, device):
    batch_size, num_nodes, _ = matrix.size()

    for i in range(batch_size):
        # 패딩되지 않은 부분의 하삼각행렬을 얻습니다.
        lower_triangular = torch.tril(matrix[i, :num_nodes, :num_nodes])

        # 패딩되지 않은 부분의 상삼각행렬을 하삼각행렬의 전치로 설정합니다.
        matrix[i, :num_nodes, :num_nodes][torch.triu_indices(num_nodes, num_nodes, offset=1)] = lower_triangular.T[torch.triu_indices(num_nodes, num_nodes, offset=1)]

    return matrix

def make_upper_follow_lower_torch(matrix, device):
    # 하삼각행렬을 얻습니다.
    lower_triangular = torch.tril(matrix)

    # 상삼각행렬을 하삼각행렬의 전치로 설정합니다.
    for i in range(matrix.size(0)):  # 배치 차원에 대해 반복
        matrix[i][torch.triu_indices(matrix.size(1), matrix.size(2), offset=1, device=device)] = lower_triangular[i].T[torch.triu_indices(matrix.size(1), matrix.size(2), offset=1, device=device)]

    return matrix

def cross_entropy_loss(pred, trg, pad_idx):
    """
    Compute the binary cross-entropy loss between predictions and targets.

    Args:
    - pred (torch.Tensor): Model predictions.
    - trg (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: Computed BCE loss.
    """
    loss = F.binary_cross_entropy(torch.sigmoid(pred[:, :-1]), get_clipped_adj_matrix(trg[:, 1:]), reduction='none')

    # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
    pad_mask = get_pad_mask(trg[:, 1:, 0], pad_idx=pad_idx)
    sub_mask = get_subsequent_mask(trg[:, :, 0])[:, 1:, :]
    mask = pad_mask.unsqueeze(-1).expand(-1, -1, loss.shape[2]) & sub_mask

    # mask 적용
    masked_loss = loss * mask.float()
    # 손실의 평균 반환
    return masked_loss.mean()

def test(sos_idx, eos_idx, pad_idx, d_street, d_unit, d_model, n_layer, n_head,
         n_building, n_boundary, dropout, checkpoint_epoch, train_ratio, val_ratio, test_ratio,
         use_global_attn, use_street_attn, use_local_attn, save_dir_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Subsequent initializations will use the already loaded full dataset
    test_dataset = GraphDataset(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                                data_type='test', load=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Initialize the Transformer model
    transformer = GraphTransformer(n_building=n_building, sos_idx=sos_idx, eos_idx=eos_idx,
                                        pad_idx=pad_idx,
                                        d_street=d_street, d_unit=d_unit, d_model=d_model,
                                        d_inner=d_model * 4, n_layer=n_layer, n_head=n_head,
                                        dropout=dropout,
                                        use_global_attn=use_global_attn,
                                        use_street_attn=use_street_attn,
                                        use_local_attn=use_local_attn).to(device=device)

    checkpoint = torch.load("./models/" + save_dir_path + "/transformer_epoch_"+ str(checkpoint_epoch) + ".pth")
    transformer.load_state_dict(checkpoint['model_state_dict'])

    transformer.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            # Get the source and target sequences from the batch
            src_unit_seq, src_street_seq, street_index_seq, trg_adj_seq, cur_n_street = data
            gt_adj_seq = trg_adj_seq.to(device=device, dtype=torch.float32)
            src_unit_seq = src_unit_seq.to(device=device, dtype=torch.float32)
            src_street_seq = src_street_seq.to(device=device, dtype=torch.float32)
            street_index_seq = street_index_seq.to(device=device, dtype=torch.long)
            trg_adj_seq = trg_adj_seq.to(device=device, dtype=torch.float32)
            cur_n_street = cur_n_street.to(device=device, dtype=torch.long)

            # Greedy Search로 시퀀스 생성
            decoder_input = trg_adj_seq[:, :1]  # 시작 토큰만 포함

            # output 값을 저장할 텐서를 미리 할당합니다.
            output_storage = torch.zeros_like(trg_adj_seq, device=device)

            for t in range(gt_adj_seq.shape[1] - 1):  # 임의의 제한값
                output = transformer(src_unit_seq, src_street_seq, street_index_seq, decoder_input, cur_n_street)
                output_storage[:, t] = output[:, t].detach()
                next_token = (torch.sigmoid(output) > 0.5).long()[:, t].unsqueeze(-2)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                decoder_input = make_upper_follow_lower_torch_padded(decoder_input, device)

            # Compute the losses using the generated sequence
            loss = cross_entropy_loss(output_storage, gt_adj_seq, pad_idx).detach().item()
            print(f"Loss CE: {loss:.4f}")
            plot(decoder_input.squeeze().detach().cpu().numpy(),
                 gt_adj_seq.squeeze().detach().cpu().numpy(),
                 idx + 1)

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
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
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--train_ratio", type=float, default=0.89, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--use_global_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_street_attn", type=bool, default=True, help="Use checkpoint index.")
    parser.add_argument("--use_local_attn", type=bool, default=True, help="Use checkpoint index.")
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

    test(sos_idx=opt.sos_idx, eos_idx=opt.eos_idx, pad_idx=opt.pad_idx,
         d_street=opt.d_street, d_unit=opt.d_unit, d_model=opt.d_model, n_layer=opt.n_layer, n_head=opt.n_head,
         n_building=opt.n_building, n_boundary=opt.n_boundary, dropout=opt.dropout, checkpoint_epoch=opt.checkpoint_epoch,
         train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
         use_global_attn=opt.use_global_attn, use_street_attn=opt.use_street_attn, use_local_attn=opt.use_local_attn,
         save_dir_path=opt.save_dir_path)