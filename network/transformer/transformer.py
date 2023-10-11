import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.transformer.layer import EncoderLayer, DecoderLayer

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
    def __init__(self, n_layer, n_head, d_k, d_v, d_model, d_inner, d_unit, d_street, dropout=0.1, n_boundary=200):
        super().__init__()

        self.pos_enc = nn.Linear(2, 1)
        self.unit_enc = nn.Linear(d_unit, d_model)
        self.street_enc = nn.Linear(d_street, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k,d_v, dropout)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_unit_seq, src_street_seq, mask):
        src_unit_seq = self.pos_enc(src_unit_seq).squeeze(dim=-1)
        src_street_seq = self.pos_enc(src_street_seq).squeeze(dim=-1)
        enc_output = self.unit_enc(src_unit_seq) + self.street_enc(src_street_seq)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, mask)

        return enc_output

class Decoder(nn.Module):
    def __init__(self, n_building, n_layer, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1, n_boundary=200):
        super().__init__()

        self.building_emb = nn.Embedding(n_building, d_model, padding_idx=pad_idx)
        self.building_enc = nn.Linear(n_building, 1)
        self.pos_enc = PositionalEncoding(d_model, n_boundary=n_boundary)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k,d_v, dropout)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, seq, enc_output, mask, trg_pad_mask):
        dec_output = self.building_emb(seq)
        trg_pad_mask = trg_pad_mask.unsqueeze(dim=-1).repeat((1, 1, 1, self.d_model))
        dec_output = dec_output * trg_pad_mask
        dec_output = dec_output.sum(dim=-2) / trg_pad_mask.sum(dim=-2)

        dec_output = self.pos_enc(dec_output)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, _, _ = dec_layer(dec_output, enc_output, mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, n_building=100, pad_idx=0, eos_idx=29, d_model=512, d_inner=2048,
                 n_layer=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_boundary=200,
                 d_unit=8, d_street=32):
        super().__init__()

        self.encoder = Encoder(n_boundary=n_boundary,
                               d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head,
                               d_k=d_k, d_v=d_v, d_unit=d_unit, d_street=d_street, dropout=dropout)
        self.decoder = Decoder(n_building=n_building, n_boundary=n_boundary,
                               d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head,
                               d_k=d_k, d_v=d_v, pad_idx=pad_idx, dropout=dropout)
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.building_fc = nn.Linear(d_model, n_building, bias=False)

    def forward(self, src_unit_seq, src_street_seq, trg_seq):
        src_pad_mask = get_pad_mask(trg_seq[:, :, 0], pad_idx=self.eos_idx).unsqueeze(-2)
        trg_pad_mask = get_pad_mask(trg_seq, pad_idx=self.pad_idx)
        sub_mask = get_subsequent_mask(trg_seq[:, :, 0])
        mask = src_pad_mask & sub_mask

        enc_output = self.encoder(src_unit_seq, src_street_seq, src_pad_mask)
        dec_output = self.decoder(trg_seq, enc_output, mask, trg_pad_mask)

        output = self.building_fc(dec_output)

        return output