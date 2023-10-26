import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Implements the scaled dot product attention mechanism.

    Args:
    - temperature (float): The scaling factor for the attention.
    - attn_dropout (float, optional): Dropout rate for the attention weights. Default is 0.1.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Compute the scaled dot product attention given a query, key, and value.

        Args:
        - q (torch.Tensor): Query tensor of shape [batch_size, num_heads, seq_len, d_k].
        - k (torch.Tensor): Key tensor of shape [batch_size, num_heads, seq_len, d_k].
        - v (torch.Tensor): Value tensor of shape [batch_size, num_heads, seq_len, d_v].
        - mask (torch.Tensor, optional): Mask tensor of shape [batch_size, 1, 1, seq_len].

        Returns:
        - tuple: Tuple containing the output tensor and the attention weights tensor.
        """
        # Compute the attention scores
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # Apply the mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Compute the attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute the output tensor by weighting the values with the attention weights
        output = torch.matmul(attn, v)

        return output, attn