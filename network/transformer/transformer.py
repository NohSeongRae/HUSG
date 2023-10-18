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
    def __init__(self, n_layer, n_head, d_k, d_v, d_model, d_inner, d_unit, d_street, dropout=0.1, n_boundary=200,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

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

    def forward(self, src_unit_seq, src_street_seq, mask, src_street_mask, src_local_mask):
        src_unit_seq = self.pos_enc(src_unit_seq).squeeze(dim=-1)
        src_street_seq = self.pos_enc(src_street_seq).squeeze(dim=-1)
        enc_output = self.unit_enc(src_unit_seq) + self.street_enc(src_street_seq)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, mask,src_street_mask,src_local_mask)

        return enc_output

class Decoder(nn.Module):
    def __init__(self, n_building, n_layer, n_head, d_k, d_v, d_model, d_inner, pad_idx, eos_idx, dropout=0.1, n_boundary=200,
                 use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

        self.building_emb = nn.Embedding(3, d_model, padding_idx=eos_idx)
        self.building_enc = nn.Linear(n_building, 1)
        self.pos_enc = PositionalEncoding(d_model, n_boundary=n_boundary)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k,d_v, dropout,
                         use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                         use_local_attn=use_local_attn)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    # mask: 일반적인 transformer 의 deocder 에서 사용되는 mask, pad mask 랑 subsequent mask 의 합
    # trg_pad_mask: 우리는 단어가 아닌, 집합이기 때문에, 집합의 크기를 맞추기 위해 사용된 pad idx 를 제외하고 aggregate 하기 위한 mask
    # trg_street_mask: 자기가 속한 street 과 동일한 street index 가진 token 에 대해서만 학습에 들어가도록 만들어주는 mask
    # trg_local_mask: 자기 포함 자기 뒤 5개를 보도록 만들었던 걸로 기억함. 만약 앞에 거 보고 싶으면, 상삼각행렬로 만들든 gpt 랑 잘 놀아보면 됨
    def forward(self, seq, enc_output, mask, trg_street_mask, trg_local_mask):
        dec_output = self.building_emb(seq)
        dec_output = self.pos_enc(dec_output)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output, mask, trg_local_mask,trg_street_mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, n_building=100, pad_idx=0, eos_idx=2, d_model=512, d_inner=2048,
                 n_layer=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_boundary=200,
                 d_unit=8, d_street=32, use_global_attn=True, use_street_attn=True, use_local_attn=True):
        super().__init__()

        self.encoder = Encoder(n_boundary=n_boundary,
                               d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head,
                               d_k=d_k, d_v=d_v, d_unit=d_unit, d_street=d_street, dropout=dropout,
                               use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                               use_local_attn=use_local_attn)
        self.decoder = Decoder(n_building=n_building, n_boundary=n_boundary, eos_idx=eos_idx,
                               d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head,
                               d_k=d_k, d_v=d_v, pad_idx=pad_idx, dropout=dropout,
                               use_global_attn=use_global_attn, use_street_attn=use_street_attn,
                               use_local_attn=use_local_attn)
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.building_fc = nn.Linear(d_model, 1, bias=False)

    def forward(self, src_unit_seq, src_street_seq, trg_building_seq, trg_street_seq):
        src_pad_mask = get_pad_mask(trg_building_seq, pad_idx=self.eos_idx).unsqueeze(-2)
        src_street_mask = get_street_mask(trg_street_seq) & src_pad_mask
        src_local_mask = get_local_mask(trg_building_seq) & src_pad_mask
        sub_mask = get_subsequent_mask(trg_building_seq)
        trg_pad_mask = src_pad_mask & sub_mask
        trg_street_mask = src_street_mask & trg_pad_mask
        trg_local_mask = src_local_mask & trg_pad_mask

        enc_output = self.encoder(src_unit_seq, src_street_seq, src_pad_mask, src_street_mask, src_local_mask)
        dec_output = self.decoder(trg_building_seq, enc_output, trg_pad_mask, trg_street_mask, trg_local_mask)

        output = self.building_fc(dec_output).squeeze(-1)

        return output