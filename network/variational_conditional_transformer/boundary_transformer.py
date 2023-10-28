import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layer import EncoderLayer, DecoderLayer

def get_street_mask(seq):
    street_mask = seq[:, :, None] == seq[:, None, :]

    return street_mask

def get_local_mask(seq):
    sz_b, len_s = seq.size()
    tril_mask1 = torch.tril(torch.ones((1, len_s, len_s), device=seq.device), diagonal=-1)
    tril_mask2 = torch.tril(torch.ones((1, len_s, len_s), device=seq.device), diagonal=-6)
    local_mask = (tril_mask1 - tril_mask2).bool()

    return local_mask

def get_pad_mask(seq, pad_idx):
    mask = (seq != pad_idx)
    return mask

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_boundary=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_boundary, d_hid))

    def _get_sinusoid_encoding_table(self, n_boundary, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_boundary)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    def __init__(self, pad_idx, n_layer, n_head, d_k, d_v, d_model, d_inner, d_unit, d_street, dropout=0.1, n_boundary=200,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

        self.pad_idx = pad_idx

        self.pos_enc = nn.Linear(2, 1)
        self.unit_enc = nn.Linear(d_unit, d_model)
        self.street_enc = nn.Linear(d_street, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k,d_v, dropout,
                         use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                         use_local_attn=use_local_attn)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_unit_seq, src_street_seq, trg_street_seq):
        enc_output = self.encoding(src_unit_seq, src_street_seq, trg_street_seq)

        return enc_output

    def encoding(self, src_unit_seq, src_street_seq, street_index_seq):
        src_pad_mask = get_pad_mask(street_index_seq, pad_idx=self.pad_idx).unsqueeze(-2)
        src_street_mask = get_street_mask(street_index_seq) & src_pad_mask
        src_local_mask = get_local_mask(street_index_seq) & src_pad_mask

        src_unit_seq = self.pos_enc(src_unit_seq).squeeze(dim=-1)
        src_street_seq = self.pos_enc(src_street_seq).squeeze(dim=-1)
        enc_output = self.unit_enc(src_unit_seq) + self.street_enc(src_street_seq)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, src_pad_mask, src_street_mask, src_local_mask)

        return enc_output

class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, d_model // 8)
        self.layer_norm1 = nn.LayerNorm(d_model // 2, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model // 4, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model // 8, eps=1e-6)
        self.d_model = d_model

    def forward(self, enc_output):
        dec_output = self.fc1(enc_output)
        dec_output = F.relu(dec_output)
        dec_output = self.layer_norm1(dec_output)

        dec_output = self.fc2(dec_output)
        dec_output = F.relu(dec_output)
        dec_output = self.layer_norm2(dec_output)

        dec_output = self.fc3(dec_output)
        dec_output = F.relu(dec_output)
        dec_output = self.layer_norm3(dec_output)

        return dec_output

class BoundaryTransformer(nn.Module):
    def __init__(self, pad_idx=0, d_model=512, d_inner=2048,
                 n_layer=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_boundary=200,
                 d_unit=8, d_street=32, use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

        self.encoder = Encoder(pad_idx=pad_idx, n_boundary=n_boundary,
                               d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head,
                               d_k=d_k, d_v=d_v, d_unit=d_unit, d_street=d_street, dropout=dropout,
                               use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                               use_local_attn=use_local_attn)
        self.decoder = Decoder(d_model=d_model)
        self.pad_idx = pad_idx
        self.fc = nn.Linear(d_model // 8, 4)

    def forward(self, src_unit_seq, src_street_seq, street_index_seq):
        enc_output = self.encoder(src_unit_seq, src_street_seq, street_index_seq)
        dec_output = self.decoder(enc_output)

        output = self.fc(dec_output)
        output = torch.sigmoid(output)

        return output