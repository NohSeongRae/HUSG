import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output = self.self_attn(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, dec_mask=None, enc_mask=None):
        dec_output = self.self_attn(dec_input, dec_input, dec_input, mask=dec_mask)
        dec_output = self.cross_attn(dec_output, enc_output, enc_output, mask=enc_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output