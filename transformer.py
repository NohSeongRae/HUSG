import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class positional_encoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):