import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch

class BoundaryMaskEncoder(nn.Module):
    def __init__(self, image_size, inner_channel, bottleneck):
        super(BoundaryMaskEncoder, self).__init__()

        self.image_size = image_size
        self.inner_channel = inner_channel

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, int(self.inner_channel / 8), 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(int(self.inner_channel / 8), int(self.inner_channel / 4), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(int(self.inner_channel / 4), int(self.inner_channel / 2), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(int(self.inner_channel / 2), int(self.inner_channel), 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
        )

        channel_num = int((image_size / 2**4)**2 * inner_channel)
        self.linear = nn.Linear(channel_num, bottleneck)

    def forward(self, mask):
        mask = mask.view(-1, 1, self.image_size, self.image_size)
        mask = self.cnn_encoder(mask)
        mask = torch.flatten(mask, 1)
        mask = self.linear(mask)
        return mask

class GraphConditionEncoder(nn.Module):
    def __init__(self, T, feature_dim, latent_dim, n_head):
        super(GraphConditionEncoder, self).__init__()

        self.street_fc = nn.Linear(128, feature_dim)

        self.convlayer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.e_conv1 = self.convlayer(feature_dim, feature_dim, heads=n_head)
        self.e_conv2 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
        self.e_conv3 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)

        self.aggregate = nn.Linear(int(feature_dim * (1.0 + n_head * T)), latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, data, edge_index):
        street_feature = data.condition_street_feature.view(-1, 128)
        street_feature = self.street_fc(street_feature)
        street_feature = F.relu(street_feature)

        n_embed_0 = street_feature

        n_embed_1 = F.relu(self.e_conv1(n_embed_0, edge_index))
        n_embed_2 = F.relu(self.e_conv2(n_embed_1, edge_index))
        n_embed_3 = F.relu(self.e_conv3(n_embed_2, edge_index))

        g_embed_0 = self.global_pool(n_embed_0, data.batch)
        g_embed_1 = self.global_pool(n_embed_1, data.batch)
        g_embed_2 = self.global_pool(n_embed_2, data.batch)
        g_embed_3 = self.global_pool(n_embed_3, data.batch)

        g_embed = torch.cat((g_embed_0, g_embed_1, g_embed_2, g_embed_3), 1)
        latent = self.aggregate(g_embed)

        return latent

class GraphEncoder(nn.Module):
    def __init__(self, T, feature_dim, latent_dim, n_head, only_building_graph):
        super(GraphEncoder, self).__init__()

        self.only_building_graph = only_building_graph

        if not only_building_graph:
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
        if not self.only_building_graph:
            street_feature = data.street_feature.view(-1, 128)
            street_feature = self.street_fc(street_feature)
            street_feature = F.relu(street_feature)

        building_feature = data.building_feature
        building_feature = self.building_fc(building_feature)
        building_feature = F.relu(building_feature)

        if not self.only_building_graph:
            n_embed_0 = street_feature * data.street_mask + building_feature * data.building_mask
        else:
            n_embed_0 = building_feature

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
    def __init__(self, feature_dim, latent_dim, n_head, bottleneck):
        super(GraphDecoder, self).__init__()

        self.dec_feature_init = nn.Linear(latent_dim + bottleneck, feature_dim)

        self.convlayer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.d_conv1 = self.convlayer(feature_dim + 180, feature_dim, heads=n_head)
        self.d_conv2 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
        self.d_conv3 = self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)

        self.dec_pos = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_pos = nn.Linear(feature_dim, 2)

        self.dec_size = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_size = nn.Linear(feature_dim, 2)

        self.dec_theta = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_theta = nn.Linear(feature_dim, 1)

    def forward(self, z, condition, edge_index, batch):
        z = torch.cat([z, condition], dim=1)
        z = self.dec_feature_init(z)
        z = z[batch]

        pos = self.node_order_within_batch(batch)
        z = torch.cat([z, pos], 1)

        d_embed_0 = F.relu(z)
        d_embed_1 = F.relu(self.d_conv1(d_embed_0, edge_index))
        d_embed_2 = F.relu(self.d_conv2(d_embed_1, edge_index))
        d_embed_3 = F.relu(self.d_conv3(d_embed_2, edge_index))

        output_pos = F.relu(self.dec_pos(d_embed_3))
        output_pos = self.fc_pos(output_pos)

        output_size = F.relu(self.dec_size(d_embed_3))
        output_size = self.fc_size(output_size)

        output_theta = F.relu(self.dec_theta(d_embed_3))
        output_theta = self.fc_theta(output_theta)

        return output_pos, output_size, output_theta

    def node_order_within_batch(self, batch):
        order_within_batch = torch.zeros_like(batch)
        unique_batches = batch.unique()
        for ub in unique_batches:
            mask = (batch == ub)
            order_within_batch[mask] = torch.arange(mask.sum(), device=batch.device)

        one_hot_order = torch.nn.functional.one_hot(order_within_batch, num_classes=180)
        return one_hot_order

class GraphCVAE(nn.Module):
    def __init__(self, T=3, feature_dim=256, latent_dim=256, n_head=8,
                 image_size=64, inner_channel=80, bottleneck=128, only_building_graph=False,
                 condition_type='graph'):
        super(GraphCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.condition_type = condition_type

        if condition_type == 'image':
            self.condition_encoder = BoundaryMaskEncoder(image_size=image_size, inner_channel=inner_channel, bottleneck=bottleneck)
        elif condition_type == 'graph':
            self.condition_encoder = GraphConditionEncoder(T=T, feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head)

        self.encoder = GraphEncoder(T=T, feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head,
                                    only_building_graph=only_building_graph)
        self.decoder = GraphDecoder(feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head, bottleneck=bottleneck)

    def reparameterize(self, mu, logvar):
        return (torch.exp(0.5 * logvar)) * (torch.randn_like(torch.exp(0.5 * logvar))) + mu

    def forward(self, data):
        print(data)
        edge_index = data.edge_index
        mu, log_var = self.encoder(data, edge_index)
        z = self.reparameterize(mu, log_var)

        if self.condition_type == 'image':
            condition = self.condition_encoder(data.condition)
        else:
            print(data.condition)
            condition = self.condition_encoder(data.condition, data.condition.edge_index)

        output_pos, output_size, output_theta = self.decoder(z, condition, edge_index, data.batch)

        return output_pos, output_size, output_theta, mu, log_var

    def test(self, data):
        z = torch.normal(mean=0, std=1, size=(1, self.latent_dim)).to(device=data.edge_index.device)

        if self.condition_type == 'image':
            condition = self.condition_encoder(data.condition)
        else:
            condition = Batch.from_data_list(data.condition)
            condition = self.condition_encoder(condition, condition.edge_index)

        output_pos, output_size, output_theta = self.decoder(z, condition, data.edge_index, data.batch)

        return output_pos, output_size, output_theta
