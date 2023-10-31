import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import degree

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

        self.convlayer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.d_conv1 = self.convlayer(feature_dim + 180, feature_dim, heads=n_head)
        self.d_conv2 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
        self.d_conv3 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)

        self.dec_geo = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_geo = nn.Linear(feature_dim, 5)

    def forward(self, z, edge_index, batch):
        z = self.dec_feature_init(z)
        z = z[batch]

        pos = self.node_order_within_batch(batch)
        z = torch.cat([z, pos], 1)

        d_embed_0 = F.relu(z)
        d_embed_1 = F.relu(self.d_conv1(d_embed_0, edge_index))
        d_embed_2 = F.relu(self.d_conv2(d_embed_1, edge_index))
        d_embed_3 = F.relu(self.d_conv3(d_embed_2, edge_index))

        output = F.relu(self.dec_geo(d_embed_3))
        output = self.fc_geo(output)

        return output

    def node_order_within_batch(self, batch):
        order_within_batch = torch.zeros_like(batch)
        unique_batches = batch.unique()
        for ub in unique_batches:
            mask = (batch == ub)
            order_within_batch[mask] = torch.arange(mask.sum(), device=batch.device)

        one_hot_order = torch.nn.functional.one_hot(order_within_batch, num_classes=180)
        return one_hot_order

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