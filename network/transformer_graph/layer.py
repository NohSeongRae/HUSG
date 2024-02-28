import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, n_head=8, dropout=0.1, use_global_attn=True):
        super(EncoderLayer, self).__init__()

        self.use_global_attn = use_global_attn

        if use_global_attn:
            self.global_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, global_mask=None, street_mask=None, local_mask=None):
        enc_output = None

        enc_global_output = self.global_attn(enc_input, enc_input, enc_input, mask=global_mask)
        enc_output = self.pos_ffn(enc_global_output)

        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, n_head=8, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.enc_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output):
        dec_output = None

        dec_output = self.enc_attn(dec_input, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)

        return dec_output