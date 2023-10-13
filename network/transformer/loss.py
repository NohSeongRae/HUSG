import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm



def cross_entropy_loss(pred, trg):
    """
    Compute the binary cross-entropy loss between predictions and targets.

    Args:
    - pred (torch.Tensor): Model predictions.
    - trg (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: Computed BCE loss.
    """

    loss = F.binary_cross_entropy_with_logits(pred, trg.float(), reduction='none')

    # pad_idx에 해당하는 레이블을 무시하기 위한 mask 생성
    mask = torch.zeros_like(trg)
    mask[:, :, 0] = 1
    mask[:, :, -1] = 1

    # mask 적용
    masked_loss = loss * mask

    # 손실의 평균 반환
    return masked_loss.mean()
