import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
import numpy as np

class MsgPass(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        tmp = self.lin(tmp)
        return tmp

class Encoder(nn.Module):
    def __init__(self, n_building=30, n_semantic=11, d_model=256, n_iter=3):
        super().__init__()

        self.n_building = n_building
        self.n_semantic = n_semantic
        self.d_model = d_model
        self.n_iter = n_iter

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.lot_geometry_encoder = nn.Linear(5, d_model)
        self.lot_semantic_embedding = nn.Embedding(n_semantic, d_model)

        self.lot_encoder = nn.Linear(d_model + d_model + n_building, d_model)

        self.node_edge_fuse_encoder_1 = MsgPass(d_model, d_model)
        self.node_edge_fuse_encoder_2 = MsgPass(d_model, d_model)
        self.node_edge_fuse_encoder_3 = MsgPass(d_model, d_model)

        self.global_pool = torch_geometric.nn.global_max_pool

        self.block_aggregate_encoder = nn.Linear(d_model * (n_iter + 1), d_model)

        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_var = nn.Linear(d_model, d_model)

    def forward(self, x):
        # n_building, geometry (5) + semantic (1) + position (n_building)

        batch_size = x.ptr.shape[0] - 1

        x_geometry = x.geometry
        x_semantic = x.semantic
        edge_index = x.edge_index

        enc_geometry = self.lot_geometry_encoder(x_geometry)
        enc_semantic = self.lot_semantic_embedding(x_semantic)

        x_position = torch.eye(self.n_building, dtype=torch.float32).to(self.device).repeat(batch_size, 1)

        enc_output = torch.cat((enc_geometry, enc_semantic, x_position), dim=1)
        enc_output = F.relu(self.lot_encoder(enc_output))
        g_embed_0 = self.global_pool(enc_output, x.batch)

        enc_output = F.relu(self.node_edge_fuse_encoder_1(enc_output, edge_index))
        g_embed_1 = self.global_pool(enc_output, x.batch)

        enc_output = F.relu(self.node_edge_fuse_encoder_2(enc_output, edge_index))
        g_embed_2 = self.global_pool(enc_output, x.batch)

        enc_output = F.relu(self.node_edge_fuse_encoder_3(enc_output, edge_index))
        g_embed_3 = self.global_pool(enc_output, x.batch)

        g_embed = torch.cat((g_embed_0, g_embed_1, g_embed_2, g_embed_3), dim=1)
        latent = self.block_aggregate_encoder(g_embed)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, n_building=30, n_semantic=11, d_model=256, n_iter=3):
        super().__init__()

        self.n_building = n_building
        self.n_semantic = n_semantic
        self.d_model = d_model
        self.n_iter = n_iter

        self.block_box_decoder_1 = nn.Linear(d_model, d_model)
        self.block_box_decoder_2 = nn.Linear(d_model, 1)

        self.edge_decoder_1 = nn.Linear((d_model + n_building) * 2, d_model)
        self.edge_decoder_2 = nn.Linear(d_model, 1)

        self.lot_inital_decoder = nn.Linear(d_model, d_model * n_building)

        self.node_edge_fuse_decoder_1 = MsgPass(d_model + n_building, d_model)
        self.node_edge_fuse_decoder_2 = MsgPass(d_model, d_model)
        self.node_edge_fuse_decoder_3 = MsgPass(d_model, d_model)

        self.node_exists_decoder = nn.Linear(d_model, 1)
        self.landuse_head_layer = nn.Linear(d_model, n_semantic)
        self.box_head_layer = nn.Linear(d_model, 5)
        self.bound_head_layer = nn.Linear(d_model, 4)
        self.merge_head_layer = nn.Linear(d_model, 2)

    def forward(self, z):
        lot_init = self.lot_inital_decoder(z)
        lot_init = F.relu(lot_init.view(-1, self.d_model))

        aspect_ratio = F.relu(self.block_box_decoder_1(lot_init))
        aspect_ratio = self.block_box_decoder_2(aspect_ratio)

        batch_size = z.shape[0]
        d_position = torch.eye(self.n_building, dtype=torch.float32).to(self.device).repeat(batch_size, 1)
        expanded_lot_init = torch.cat((lot_init, d_position), dim=1)
        expanded_lot_init = expanded_lot_init.unsqueeze(2).expand(-1, -1, self.n_building, -1)
        expanded_lot_init = F.relu(self.edge_decoder_1(expanded_lot_init))
        adj_matrix = F.sigmoid(self.edge_decoder_2(expanded_lot_init).squeeze(-1))
        threshold = 0.5
        binary_adj_matrix = (adj_matrix >= threshold)
        edge_index = self.generate_edge_index(binary_adj_matrix, batch_size)

        d_embed_0 = torch.cat((lot_init, d_position), dim=1)
        d_embed_1 = F.relu(self.node_edge_fuse_decoder_1(d_embed_0, edge_index))
        d_embed_2 = F.relu(self.node_edge_fuse_decoder_2(d_embed_1, edge_index))
        d_embed_3 = F.relu(self.node_edge_fuse_decoder_3(d_embed_2, edge_index))

        lot_exists_prob = F.sigmoid(self.node_exists_decoder(d_embed_3))
        land_use_attribute = F.softmax(self.landuse_head_layer(d_embed_3), dim=-1)
        lot_geometry = self.box_head_layer(d_embed_3)
        lot_boundary_attribute = self.bound_head_layer(d_embed_3)
        lot_merge_operation = self.merge_head_layer(d_embed_3)

        return (lot_exists_prob, lot_geometry, land_use_attribute, lot_boundary_attribute, lot_merge_operation,
                adj_matrix, aspect_ratio)

    def generate_edge_index(self, binary_adj_matrix, batch_size):
        edge_indices = []

        # Create edge_index for each graph in the batch
        for b in range(batch_size):
            adj_matrix = binary_adj_matrix[b]
            edge_index = torch.stack(torch.where(adj_matrix))

            # Offset node indices by the number of nodes in previous graphs
            edge_index[0, :] += b * self.n_building

            edge_indices.append(edge_index)

        # Concatenate to obtain final edge_index tensor
        final_edge_index = torch.cat(edge_indices, dim=1)
        return final_edge_index

class BlockPlanner(nn.Module):
    def __init__(self, n_building=30, n_semantic=11, d_model=256, n_iter=3):
        super().__init__()

        self.n_building = n_building
        self.n_semantic = n_semantic
        self.d_model = d_model
        self.n_iter = n_iter

        self.encoder = Encoder(n_building=n_building, n_semantic=n_semantic, d_model=d_model, n_iter=n_iter)
        self.decoder = Decoder(n_building=n_building, n_semantic=n_semantic, d_model=d_model, n_iter=n_iter)

    def forward(self, x):
        mu, log_var = self.encoder(x)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        (lot_exists_prob, lot_geometry, land_use_attribute, lot_boundary_attribute, lot_merge_operation,
         adj_matrix, aspect_ratio) = self.decoder(z)

        outputs = dict()
        outputs['lot_exists_prob'] = lot_exists_prob
        outputs['lot_geometry'] = lot_geometry
        outputs['land_use_attribute'] = land_use_attribute
        outputs['lot_boundary_attribute'] = lot_boundary_attribute
        outputs['lot_merge_operation'] = lot_merge_operation
        outputs['adj_matrix'] = adj_matrix
        outputs['aspect_ratio'] = aspect_ratio
        outputs['mu'] = mu
        outputs['log_var'] = log_var

        return outputs