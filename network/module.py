import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        """
        Initializes the ScaledDotProductAttention layer.

        Parameters:
        - temperature (float): A scaling factor to normalize the dot products of the query and key vectors. Often set to the square root of the dimension of the key vectors.
        - attn_dropout (float, optional): Dropout rate for the attention weights. Default is 0.1.
        """

        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for the ScaledDotProductAttention layer.

        Parameters:
        - q (Tensor): The query vectors. Shape: (batch_size, num_queries, query_length, d_k), where d_k is the dimension of the query vectors.
        - k (Tensor): The key vectors. Shape should match the query vectors.
        - v (Tensor): The value vectors. Shape: (batch_size, num_values, value_length, d_v), where d_v is the dimension of the value vectors.
        - mask (Tensor, optional): An optional mask to prevent attention to certain positions. A value of 0 in the mask indicates that the corresponding position should not be attended to.

        Returns:
        - output (Tensor): The result of applying attention to the value vectors based on the scaled dot products of query and key vectors. Shape: (batch_size, num_queries, query_length, d_v).
        - attn (Tensor): The attention weights after applying the softmax function. Shape: (batch_size, num_queries, query_length, value_length).
        """

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn