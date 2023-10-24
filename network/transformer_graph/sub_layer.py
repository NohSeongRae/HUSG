import torch
import torch.nn as nn
from module import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_model // n_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_model // self.n_head, self.d_model // self.n_head, self.n_head
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

    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_inner)
        self.w2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w1(x)
        x = torch.relu(x)
        x = self.w2(x)
        x += residual
        x = self.layer_norm(x)

        return x