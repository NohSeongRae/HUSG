import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        """
        Initializes the EncoderLayer.

        Parameters:
        - d_model (int): The dimension of the input and output vectors.
        - d_inner (int): The dimension of the inner layer in the position-wise feed-forward network.
        - n_head (int): The number of heads in the multi-head attention mechanism.
        - dropout (float): The dropout rate.
        """

        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        """
        Forward pass of the encoder layer.

        Parameters:
        - enc_input (Tensor): The input tensor to the encoder layer.
        - mask (Tensor, optional): The mask for the multi-head attention mechanism.

        Returns:
        - Tensor: The output tensor of the encoder layer after applying self-attention and position-wise feed-forward network.
        """

        enc_output = self.self_attn(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        """
        Initializes the DecoderLayer.

        Parameters are the same as for the EncoderLayer.
        """

        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, dec_mask=None, enc_mask=None):
        """
        Forward pass of the decoder layer.

        Parameters:
        - dec_input (Tensor): The input tensor to the decoder layer.
        - enc_output (Tensor): The output tensor from the encoder.
        - dec_mask (Tensor, optional): The mask for the multi-head self-attention mechanism in the decoder.
        - enc_mask (Tensor, optional): The mask for the multi-head attention mechanism over the encoder's output.

        Returns:
        - Tensor: The output tensor of the decoder layer after applying self-attention, encoder-decoder attention, and position-wise feed-forward network.
        """

        dec_output = self.self_attn(dec_input, dec_input, dec_input, mask=dec_mask)
        dec_output = self.cross_attn(dec_output, enc_output, enc_output, mask=enc_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output