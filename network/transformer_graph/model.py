import torch
import torch.nn as nn
import numpy as np

from layer import EncoderLayer, DecoderLayer

def get_src_street_mask(seq):
    street_mask = seq[:, :, None] == seq[:, None, :]
    return street_mask

def get_src_local_mask(seq):
    sz_b, len_s = seq.size()
    tril_mask1 = torch.tril(torch.ones((1, len_s, len_s), device=seq.device), diagonal=2)
    tril_mask2 = torch.tril(torch.ones((1, len_s, len_s), device=seq.device), diagonal=-2)
    local_mask = (tril_mask1 - tril_mask2).bool()
    return local_mask

def get_cliped_adj_matrix(adj_matrix):
    adj_matrix[adj_matrix >= 2] = 0
    return adj_matrix

def get_trg_street_mask(adj_matrix, n_street_node):
    adj_matrix = get_cliped_adj_matrix(adj_matrix)
    batch_size, n_node1, _ = adj_matrix.size()

    # Expand dimensions to use broadcasting
    n_street_node = n_street_node.view(batch_size, 1, 1).expand(-1, -1, n_node1)
    indices = torch.arange(n_node1, device=n_street_node.device).view(1, n_node1, 1).expand(batch_size, -1, -1)

    # Create a boolean mask for core nodes
    street_mask = (indices < n_street_node + 1) & (1 < indices)

    # Calculate the mask using matrix multiplication and broadcasting
    street_mask = torch.bmm(adj_matrix[:, :n_node1, :n_node1] * street_mask.float(),
                            adj_matrix[:, :n_node1, :n_node1])

    # Clip values greater than 1
    street_mask = torch.clamp(street_mask, max=1)

    return street_mask.bool()

def get_trg_local_mask(adj_matrix):
    adj_matrix = get_cliped_adj_matrix(adj_matrix)
    return adj_matrix.bool()

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

    def _get_sinusoid_encoding_table(self, n_boundary, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_boundary)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class BoundaryEncoder(nn.Module):
    def __init__(self, n_layer=6, n_head=8, d_model=512, d_inner=2048, d_unit=8, d_street=64,
                 dropout=0.1, use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super(BoundaryEncoder, self).__init__()

        self.pos_enc = nn.Linear(2, 1)
        self.unit_enc = nn.Linear(d_unit, d_model)
        self.street_enc = nn.Linear(d_street, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout,
                         use_global_attn=use_global_attn, use_street_attn=use_street_attn, use_local_attn=use_local_attn)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_unit_seq, src_street_seq, global_mask, street_mask, local_mask):
        src_unit_seq = self.pos_enc(src_unit_seq).squeeze(dim=-1)
        src_street_seq = self.pos_enc(src_street_seq).squeeze(dim=-1)

        enc_output = self.unit_enc(src_unit_seq) + self.street_enc(src_street_seq)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, global_mask, street_mask, local_mask)

        return enc_output

class GraphDecoder(nn.Module):
    def __init__(self, n_layer=6, n_head=8, n_building=120, n_street=50, d_model=512, d_inner=2048, dropout=0.1,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super(GraphDecoder, self).__init__()

        self.node_enc = nn.Linear(n_building + n_street, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_building=n_building + n_street)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
           DecoderLayer(d_model, d_inner, n_head, dropout,
                        use_global_attn=use_global_attn, use_street_attn=use_street_attn, use_local_attn=use_local_attn)
           for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, dec_input, enc_output, global_mask, street_mask, local_mask, enc_mask):
        dec_output = self.node_enc(dec_input)
        dec_output = self.pos_enc(dec_output)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, global_mask, street_mask, local_mask, enc_mask)

        return dec_output

class GraphTransformer(nn.Module):
    def __init__(self, n_building=120, n_street=50, d_model=512, d_inner=2048, sos_idx=2, eos_idx=3, pad_idx=4,
                 n_layer=6, n_head=8, dropout=0.1, d_unit=8, d_street=64,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True, local_rank=0):
        super(GraphTransformer, self).__init__()

        self.sos_idx = sos_idx    # [2, 2, 2, ..., 2]
        self.eos_idx = eos_idx    # [3, 3, 3, ..., 3]
        self.pad_idx = pad_idx    # [4, 4, 4, ..., 4]

        self.encoder = BoundaryEncoder(n_layer=n_layer, n_head=n_head, d_model=d_model, d_inner=d_inner,
                                       d_unit=d_unit, d_street=d_street, dropout=0.1,
                                       use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                                       use_local_attn=use_local_attn)
        self.decoder = GraphDecoder(n_layer=n_layer, n_head=n_head, n_building=n_building,
                                    d_model=d_model, d_inner=d_inner, dropout=0.1,
                                    use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                                    use_local_attn=use_local_attn)

        self.dropout = nn.Dropout(dropout)
        self.adj_fc = nn.Linear(d_model, n_building + n_street)

    def forward(self, src_unit_seq, src_street_seq, street_index_seq, trg_adj_seq, n_street_node):
        src_global_mask = get_pad_mask(street_index_seq, pad_idx=0).unsqueeze(-2)
        src_street_mask = get_src_street_mask(street_index_seq) & src_global_mask
        src_local_mask = get_src_local_mask(street_index_seq) & src_global_mask

        enc_output = self.encoder(src_unit_seq, src_street_seq, src_global_mask, src_street_mask, src_local_mask)

        trg_sub_mask = get_subsequent_mask(trg_adj_seq[:, :, 0])
        trg_global_mask = get_pad_mask(trg_adj_seq[:, :, 0], pad_idx=self.pad_idx).unsqueeze(-2) & trg_sub_mask
        trg_street_mask = get_trg_street_mask(trg_adj_seq, n_street_node) & trg_global_mask
        trg_local_mask = get_trg_local_mask(trg_adj_seq) & trg_global_mask

        trg_adj_seq = trg_adj_seq * trg_sub_mask.expand(trg_adj_seq.shape[0], -1, -1)

        print(src_global_mask.shape, src_street_mask.shape, src_local_mask.shape)
        print(trg_sub_mask.shape, trg_global_mask.shape, trg_street_mask.shape, trg_local_mask.shape)

        dec_output = self.decoder(trg_adj_seq, enc_output, trg_global_mask, trg_street_mask, trg_local_mask, src_global_mask)

        output = self.dropout(dec_output)
        output = self.adj_fc(output)

        return output