import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import degree
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_node=180):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_node, d_hid))

    def _get_sinusoid_encoding_table(self, n_node, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_node)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x].clone().detach()

class GraphEncoder(nn.Module):
    def __init__(self, T, feature_dim, latent_dim, n_head):
        super(GraphEncoder, self).__init__()

        self.street_fc = nn.Linear(128, feature_dim)
        self.building_fc = nn.Linear(5, feature_dim)

        self.convlayer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.e_conv1 = self.convlayer(feature_dim, feature_dim, heads=n_head)
        self.e_conv2 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
        self.e_conv3 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)

        self.aggregate = nn.Linear(int(feature_dim * (1.0 + n_head * T)), latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, data, edge_index):
        street_feature = data.street_feature.view(-1, 128)
        building_feature = data.building_feature

        street_feature = self.street_fc(street_feature)
        building_feature = self.building_fc(building_feature)

        street_feature = F.relu(street_feature)
        building_feature = F.relu(building_feature)

        n_embed_0 = street_feature * data.street_mask + building_feature * data.building_mask
        n_embed_1 = F.relu(self.e_conv1(n_embed_0, edge_index))
        n_embed_2 = F.relu(self.e_conv2(n_embed_1, edge_index))
        n_embed_3 = F.relu(self.e_conv3(n_embed_2, edge_index))

        g_embed_0 = self.global_pool(n_embed_0, data.batch)
        g_embed_1 = self.global_pool(n_embed_1, data.batch)
        g_embed_2 = self.global_pool(n_embed_2, data.batch)
        g_embed_3 = self.global_pool(n_embed_3, data.batch)

        g_embed = torch.cat((g_embed_0, g_embed_1, g_embed_2, g_embed_3), 1)
        latent = self.aggregate(g_embed)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return mu, log_var


class GraphDecoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, n_head):
        super(GraphDecoder, self).__init__()

        self.dec_feature_init = nn.Linear(latent_dim, feature_dim)
        self.pos_enc = PositionalEncoding(feature_dim, n_node=180)

        self.convlayer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.d_conv1 = self.convlayer(feature_dim, feature_dim, heads=n_head)
        self.d_conv2 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
        self.d_conv3 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)

        self.dec_geo = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_geo = nn.Linear(feature_dim, 5)

    def forward(self, z, edge_index, batch):
        print(z.shape)
        z = self.dec_feature_init(z)
        print(z.shape)
        z = z[batch]
        print(z.shape)

        pos = self.node_order_within_batch(batch)
        print(pos.shape)
        pos = self.pos_enc(pos)
        print(pos.shape)
        z = torch.cat([z, pos], 1)
        print(z.shape)

        d_embed_0 = F.relu(z)
        d_embed_1 = F.relu(self.d_conv1(d_embed_0, edge_index))
        d_embed_2 = F.relu(self.d_conv2(d_embed_1, edge_index))
        d_embed_3 = F.relu(self.d_conv3(d_embed_2, edge_index))

        output = F.relu(self.dec_geo(d_embed_3))
        output = self.fc_geo(output)

        return output

    def node_order_within_batch(self, batch):
        num_nodes_per_graph = degree(batch, dtype=torch.long)
        cum_nodes_per_graph = torch.cat([torch.tensor([0]).to(device=batch.device),
                                         torch.cumsum(num_nodes_per_graph, dim=0)[:-1]], dim=0)
        order_within_batch = batch - cum_nodes_per_graph[batch]
        return order_within_batch

class GraphCVAE(nn.Module):
    def __init__(self, T=3, feature_dim=256, latent_dim=256, n_head=8):
        super(GraphCVAE, self).__init__()

        self.encoder = GraphEncoder(T=T, feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head)
        self.decoder = GraphDecoder(feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head)

    def reparameterize(self, mu, logvar):
        return (torch.exp(0.5 * logvar)) * (torch.randn_like(torch.exp(0.5 * logvar))) + mu

    def forward(self, data):
        edge_index = data.edge_index
        mu, log_var = self.encoder(data, edge_index)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z, edge_index, data.batch)

        return output, mu, log_var