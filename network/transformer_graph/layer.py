import torch.nn as nn
from sub_layer import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, n_head=8, dropout=0.1,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super(EncoderLayer, self).__init__()

        self.use_global_attn = use_global_attn
        self.use_street_attn = use_street_attn
        self.use_local_attn = use_local_attn

        if use_global_attn:
            self.global_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        if use_street_attn:
            self.street_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        if use_local_attn:
            self.local_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, global_mask=None, street_mask=None, local_mask=None):
        enc_output = None

        if self.use_global_attn:
            enc_global_output = self.global_attn(enc_input, enc_input, enc_input, mask=global_mask)
            if enc_output == None:
                enc_output = enc_global_output

        if self.use_street_attn:
            enc_street_output = self.street_attn(enc_input, enc_input, enc_input, mask=street_mask)
            if enc_output == None:
                enc_output = enc_street_output
            else:
                enc_output += enc_street_output

        if self.use_local_attn:
            enc_local_output = self.local_attn(enc_input, enc_input, enc_input, mask=local_mask)
            if enc_output == None:
                enc_output = enc_local_output
            else:
                enc_output += enc_local_output

        enc_output = self.pos_ffn(enc_output)

        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, n_head=8, dropout=0.1,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super(DecoderLayer, self).__init__()

        self.use_global_attn = use_global_attn
        self.use_street_attn = use_street_attn
        self.use_local_attn = use_local_attn

        if use_global_attn:
            self.global_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        if use_street_attn:
            self.street_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        if use_local_attn:
            self.local_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)

        self.enc_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, global_mask=None, street_mask=None, local_mask=None, enc_mask=None):
        dec_output = None

        if self.use_global_attn:
            dec_global_output = self.global_attn(dec_input, dec_input, dec_input, mask=global_mask)
            if dec_output == None:
                dec_output = dec_global_output

        if self.use_street_attn:
            dec_street_output = self.street_attn(dec_input, dec_input, dec_input, mask=street_mask)
            if dec_output == None:
                dec_output = dec_street_output
            else:
                dec_output += dec_street_output

        if self.use_local_attn:
            dec_local_output = self.local_attn(dec_input, dec_input, dec_input, mask=local_mask)
            if dec_output == None:
                dec_output = dec_local_output
            else:
                dec_output += dec_local_output

        dec_output = self.enc_attn(dec_output, enc_output, enc_output, mask=enc_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output