import torch
import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    Encoder layer for the Transformer model.

    Args:
    - d_model (int): Dimension of the model.
    - d_inner (int): Dimension of the inner layer in the feed forward network.
    - n_head (int): Number of multi-head attentions.
    - d_k (int): Dimension of the key.
    - d_v (int): Dimension of the value.
    - dropout (float, optional): Dropout rate. Default is 0.1.
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.use_global_attn = use_global_attn
        self.use_street_attn = use_street_attn
        self.use_local_attn = use_local_attn

        if use_global_attn:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        if use_street_attn:
            self.street_attn=MultiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        if use_local_attn:
            self.local_attn=MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None,street_attn_mask=None, local_attn_mask=None):
        """
        Forward pass for the Encoder layer.

        Args:
        - enc_input (torch.Tensor): Input tensor for the encoder.
        - slf_attn_mask (torch.Tensor, optional): Mask for the self attention mechanism.

        Returns:
        - tuple: Tuple containing the encoded output and self attention tensor.
        """
        enc_output = torch.zeros(enc_input.shape).to(device=self.device)

        if self.use_global_attn:
            enc_slf_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask
            )
            enc_output += enc_slf_output

        if self.use_street_attn:
            enc_street_output, enc_street_attn=self.slf_attn(
                enc_input, enc_input, enc_input, mask=street_attn_mask
            )
            enc_output += enc_street_output
        if self.use_local_attn:
            enc_local_output, enc_local_attn=self.slf_attn(
                enc_input, enc_input, enc_input, mask=local_attn_mask
            )
            enc_output += enc_local_output

        enc_output = self.pos_ffn(enc_output)

        return enc_output

class DecoderLayer(nn.Module):
    """
    Decoder layer for the Transformer model.

    Args:
    - d_model (int): Dimension of the model.
    - d_inner (int): Dimension of the inner layer in the feed forward network.
    - n_head (int): Number of multi-head attentions.
    - d_k (int): Dimension of the key.
    - d_v (int): Dimension of the value.
    - dropout (float, optional): Dropout rate. Default is 0.1.
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

        # Set the device for training (either GPU or CPU based on availability)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.use_global_attn = use_global_attn
        self.use_street_attn = use_street_attn
        self.use_local_attn = use_local_attn

        if use_global_attn:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        if use_street_attn:
            self.street_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        if use_local_attn:
            self.local_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, street_attn_mask=None, local_attn_mask=None):
        """
        Forward pass for the Decoder layer.

        Args:
        - dec_input (torch.Tensor): Input tensor for the decoder.
        - enc_output (torch.Tensor): Output tensor from the encoder.
        - slf_attn_mask (torch.Tensor, optional): Mask for the self attention mechanism.

        Returns:
        - tuple: Tuple containing the decoded output, decoder self attention tensor, and encoder attention tensor.
        """
        dec_output = torch.zeros(dec_input.shape).to(device=self.device)

        # global attention
        if self.use_global_attn:
            dec_slf_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=slf_attn_mask
            )
            dec_output += dec_slf_output

        # street attention
        if self.use_street_attn:
            dec_street_output, dec_street_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=street_attn_mask
            )
            dec_output += dec_street_output

        # local attention
        if self.use_local_attn:
            dec_local_output, dec_local_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=local_attn_mask
            )
            dec_output += dec_local_output

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=slf_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)

        return dec_output
