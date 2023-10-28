import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layer import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    mask = (seq != pad_idx)
    return mask

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_building=120):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_building, d_hid))

    def _get_sinusoid_encoding_table(self, n_building, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_building)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, n_x):
        return self.pos_table[:, :n_x].clone().detach()

class BuildingEncoder(nn.Module):
    def __init__(self, n_layer, n_head, n_building, d_model, d_inner, d_feature, d_k, d_v, dropout=0.1):
        super().__init__()

        self.feature_enc = nn.Linear(d_feature, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_building)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout, d_k, d_v,
                         use_global_attn=True, use_street_attn=False, use_local_attn=False)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.n_building = n_building

    def forward(self, src_building_seq, src_mask):
        enc_output = self.feature_enc(src_building_seq) + self.pos_enc(self.n_building)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, src_mask, None, None)

        mu = enc_output[:, 0, :]
        log_var = enc_output[:, 1, :]

        return mu, log_var

class BuildingDecoder(nn.Module):
    def __init__(self, n_building, n_layer, n_head, d_model, d_inner, d_k, d_v, dropout=0.1):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model, n_building=n_building)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout, d_k, d_v,
                         use_global_attn=True, use_street_attn=False, use_local_attn=False)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.building = n_building

    def forward(self, boundary_z, building_z, mask):
        pos_output = self.pos_enc(self.n_building)
        dec_output = self.dropout(pos_output)
        dec_output = self.layer_norm(dec_output)

        enc_output = torch.cat([boundary_z, building_z], dim=1)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, mask, None, None, mask)

        return dec_output

class BuildingTransformer(nn.Module):
    def __init__(self, n_building=120, pad_idx=0, d_model=512, d_inner=2048,
                 n_layer=6, n_head=8, dropout=0.1, d_feature=5, d_k=64, d_v=64):
        super().__init__()

        self.encoder = BuildingEncoder(d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_building=n_building,
                                       n_head=n_head, dropout=dropout, d_feature=d_feature, d_k=d_k, d_v=d_v)
        self.decoder = BuildingDecoder(n_building=n_building, d_model=d_model, d_inner=d_inner,
                                       n_layer=n_layer, n_head=n_head, dropout=dropout, d_k=d_k, d_v=d_v)
        self.pad_idx = pad_idx
        self.building_fc = nn.Linear(d_model, d_feature, bias=False)

    def forward(self, boundary_z, src_building_seq):
        pad_mask = get_pad_mask(src_building_seq[:, :, 0], pad_idx=0).unsqueeze(-2)

        mu, log_var = self.encoder(src_building_seq, pad_mask)
        enc_output = self.reparameterize(mu, log_var)
        dec_output = self.decoder(boundary_z, enc_output, pad_mask)

        output = self.building_fc(dec_output).squeeze(-1)
        return output, mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoding(self, boundary_z, building_z):
        dec_output = self.decoder(boundary_z, building_z, mask=None)

        output = self.building_fc(dec_output).squeeze(-1)
        return output