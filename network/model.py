import torch
import torch.nn as nn
import numpy as np

from layer import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    mask = (seq != pad_idx)
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_node):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_node, d_hid))

    def _get_sinusoid_encoding_table(self, n_boundary, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_boundary)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()

class BoundaryEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(BoundaryEncoder, self).__init__()

        n_node = 200
        self.pos_enc = PositionalEncoding(d_model, n_node=n_node)
        self.enc_init = nn.Linear(n_node + d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, boundary, enc_mask):
        enc_input = torch.cat([boundary, self.pos_enc(boundary).expand(boundary.shape[0], -1, -1)], dim=2)
        enc_input = self.enc_init(enc_input)
        enc_output = self.dropout(enc_input)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_mask)

        return enc_output

class BuildingDncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout):
        super(BuildingDncoder, self).__init__()

        n_node = 120
        self.pos_enc = PositionalEncoding(d_model, n_node=n_node)
        self.dec_init = nn.Linear(n_node + d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, building, boundary, dec_mask, enc_mask):
        dec_input = torch.cat([building, self.pos_enc(building).expand(boundary.shape[0], -1, -1)], dim=2)
        dec_input = self.dec_init(dec_input)
        dec_output = self.dropout(dec_input)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, boundary, dec_mask, enc_mask)

        return dec_output

class GraphTransformer(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout):
        super(GraphTransformer, self).__init__()

        self.encoder = BoundaryEncoder(n_layer, n_head, d_model, d_inner, dropout)
        self.decoder = BuildingDncoder(n_layer, n_head, d_model, d_inner, dropout)

        self.dropout = nn.Dropout(dropout)
        self.adj_fc = nn.Linear(d_model, 200)

    def forward(self, building, boundary, building_pad_mask, boundary_pad_mask):
        enc_output = self.encoder(boundary, boundary_pad_mask)
        dec_output = self.decoder(building, enc_output, building_pad_mask, boundary_pad_mask)

        dec_output = self.adj_fc(dec_output)
        dec_output = torch.sigmoid(dec_output)

        return dec_output
