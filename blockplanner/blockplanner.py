import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

#variant of BlockPlanner (2021 ICCV)
class BpEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, N):
        super(BpEncoder, self).__init__()
        self.lot_geometry_enc=nn.Linear(input_dim, hidden_dim)
        self.lot_semantic_enc=nn.Linear(input_dim, hidden_dim)  #CLIP_text(txt)
        self.lot_encoder=nn.Linear(hidden_dim*2+N, hidden_dim)
        self.GNN=pyg_nn.GeneralConv(hidden_dim, hidden_dim)
        self.block_aggregate_encoder=nn.Linear(hidden_dim*4, hidden_dim)

    def forward(self, data):
        x, edge_index=data.x, data.edge_index
        h_g=self.lot_geometry_enc(x)
        h_s=self.lot_semantic_enc(x)
        h=torch.cat(h_s, h_g)
        h=self.lot_encoder(h)

        h=nn.ReLU(self.GNN(x, edge_index))