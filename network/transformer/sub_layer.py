import torch
import torch.nn as nn
import torch.nn.functional as F
from network.transformer.module import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism.

    Args:
    - n_head (int): Number of parallel attention heads.
    - d_model (int): Dimensionality of the model.
    - d_k (int): Dimensionality of the key vectors.
    - d_v (int): Dimensionality of the value vectors.
    - dropout (float, optional): Dropout rate for the attention weights. Default is 0.1.
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        Compute the multi-head attention output given query, key, and value tensors.

        Args:
        - q (torch.Tensor): Query tensor.
        - k (torch.Tensor): Key tensor.
        - v (torch.Tensor): Value tensor.
        - mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions.

        Returns:
        - tuple: Tuple containing the attention output and the attention weights.
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q = self.dropout(q)
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward sub-layer.

    Args:
    - d_in (int): Input dimensionality.
    - d_hid (int): Dimensionality of the hidden layer.
    - dropout (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Compute the feed-forward network output for the given input tensor `x`.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after the position-wise feed-forward network.
        """
        residual = x
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x += residual
        x = self.layer_norm(x)

        return x