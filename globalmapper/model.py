import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch
import torchvision.models as models

class ResNet34(nn.Module):
    def __init__(self, bottleneck):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, bottleneck)

    def forward(self, mask):
        mask = mask.view(-1, 3, 224, 224)
        return self.model(mask)

class BoundaryMaskEncoder(nn.Module):
    def __init__(self, image_size, inner_channel, bottleneck):
        super(BoundaryMaskEncoder, self).__init__()

        self.image_size = image_size
        self.inner_channel = inner_channel

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(2, int(self.inner_channel / 8), 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(int(self.inner_channel / 8), int(self.inner_channel / 4), 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(int(self.inner_channel / 4), int(self.inner_channel / 2), 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(int(self.inner_channel / 2), int(self.inner_channel), 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        channel_num = int((image_size / 2 ** 4) ** 2 * inner_channel)
        self.linear = nn.Linear(channel_num, bottleneck)

    def forward(self, mask):
        mask = mask.view(-1, 2, self.image_size, self.image_size)
        mask = self.cnn_encoder(mask)
        mask = torch.flatten(mask, 1)
        mask = self.linear(mask)
        return mask


class GraphConditionEncoder(nn.Module):
    def __init__(self, T, feature_dim, bottleneck, n_head, convlayer):
        super(GraphConditionEncoder, self).__init__()

        self.bbox_fc = nn.Linear(5, feature_dim)
        if convlayer == 'gat':
            self.convlayer = torch_geometric.nn.GATConv
        elif convlayer == 'gcn':
            self.convlayer = torch_geometric.nn.GCNConv
        elif convlayer == 'gin':
            self.convlayer = lambda in_channels, out_channels: torch_geometric.nn.GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            )

        self.global_pool = torch_geometric.nn.global_max_pool

        if convlayer == 'gat':
            self.e_conv1 = self.convlayer(feature_dim, feature_dim, heads=n_head)
            self.layer_stack = nn.ModuleList([
                self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
                for _ in range(T -1)
            ])
            self.aggregate = nn.Linear(int(feature_dim * (1.0 + n_head * T)), bottleneck)
        else:
            self.e_conv1 = self.convlayer(feature_dim, feature_dim)
            self.layer_stack = nn.ModuleList([
                self.convlayer(feature_dim, feature_dim)
                for _ in range(T - 1)
            ])
            self.aggregate = nn.Linear(int(feature_dim * (1.0 + T)), bottleneck)

    def forward(self, data, edge_index):
        street_feature = data.condition_street_feature
        street_feature = self.bbox_fc(street_feature)
        street_feature = F.relu(street_feature)

        n_embed_0 = street_feature
        g_embed_0 = self.global_pool(n_embed_0, data.batch)

        n_embed_t = F.relu(self.e_conv1(n_embed_0, edge_index))
        g_embed_t = self.global_pool(n_embed_t, data.batch)

        g_embed = torch.cat((g_embed_0, g_embed_t), dim=1)

        for e_conv_t in self.layer_stack:
            n_embed_t = F.relu(e_conv_t(n_embed_t, edge_index))
            g_embed_t = self.global_pool(n_embed_t, data.batch)

            g_embed = torch.cat((g_embed, g_embed_t), dim=1)

        latent = self.aggregate(g_embed)
        return latent


class GraphEncoder(nn.Module):
    def __init__(self, T, feature_dim, latent_dim, n_head, convlayer, batch_size):
        super(GraphEncoder, self).__init__()

        self.batch_size = batch_size

        self.pos_fc = nn.Linear(2, feature_dim // 2)
        self.size_fc = nn.Linear(2, feature_dim // 2)
        # self.shape_fc = nn.Embedding(6, feature_dim // 4)
        # self.iou_fc = nn.Linear(1, feature_dim // 4)

        self.N = 120
        self.exist_embed = nn.Embedding(2, feature_dim // 2)
        self.exist_enc = nn.Linear(feature_dim // 2 + self.N, feature_dim)

        self.node_fc = nn.Linear(feature_dim * 2, feature_dim)

        if convlayer == 'gat':
            self.convlayer = torch_geometric.nn.GATConv
        elif convlayer == 'gcn':
            self.convlayer = torch_geometric.nn.GCNConv
        elif convlayer == 'gin':
            self.convlayer = lambda in_channels, out_channels: torch_geometric.nn.GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            )

        self.global_pool = torch_geometric.nn.global_max_pool

        if convlayer == 'gat':
            self.e_conv1 = self.convlayer(feature_dim, feature_dim, heads=n_head)
            self.layer_stack = nn.ModuleList([
                self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
                for _ in range(T -1)
            ])
            self.aggregate = nn.Linear(int(feature_dim * (1.0 + n_head * T)), latent_dim)
        else:
            self.e_conv1 = self.convlayer(feature_dim, feature_dim)
            self.layer_stack = nn.ModuleList([
                self.convlayer(feature_dim, feature_dim)
                for _ in range(T - 1)
            ])
            self.aggregate = nn.Linear(int(feature_dim * (1.0 + T)), latent_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, data, edge_index):
        pos_feature = data.pos_features
        pos_feature = F.relu(self.pos_fc(pos_feature))

        size_feature = data.size_features
        size_feature = F.relu(self.size_fc(size_feature))

        # shape_feature = data.shape_features
        # shape_feature = F.relu(self.shape_fc(shape_feature))
        #
        # iou_feature = data.iou_features.unsqueeze(-1)
        # iou_feature = F.relu(self.iou_fc(iou_feature))

        node_exist = data.exist_features
        node_exist = F.relu(self.exist_embed(node_exist))

        one_hot = self.node_order_within_batch(data.batch)
        node_exist = F.relu(self.exist_enc(torch.cat([node_exist, one_hot], dim=1)))

        node_feature = F.relu(self.node_fc(torch.cat([pos_feature, size_feature, node_exist], dim=1)))

        n_embed_0 = node_feature
        g_embed_0 = self.global_pool(n_embed_0, data.batch)

        n_embed_t = F.relu(self.e_conv1(n_embed_0, edge_index))
        g_embed_t = self.global_pool(n_embed_t, data.batch)

        g_embed = torch.cat((g_embed_0, g_embed_t), dim=1)

        for e_conv_t in self.layer_stack:
            n_embed_t = F.relu(e_conv_t(n_embed_t, edge_index))
            g_embed_t = self.global_pool(n_embed_t, data.batch)

            g_embed = torch.cat((g_embed, g_embed_t), dim=1)

        latent = self.aggregate(g_embed)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)

        return mu, log_var

    def node_order_within_batch(self, batch):
        order_within_batch = torch.zeros_like(batch)
        unique_batches = batch.unique()
        for ub in unique_batches:
            mask = (batch == ub)
            order_within_batch[mask] = torch.arange(mask.sum(), device=batch.device)

        one_hot_order = torch.nn.functional.one_hot(order_within_batch, num_classes=120)
        return one_hot_order
class GraphDecoder(nn.Module):
    def __init__(self, T, feature_dim, latent_dim, n_head, bottleneck, convlayer, batch_size):
        super(GraphDecoder, self).__init__()

        self.N = 120
        self.batch_size = batch_size

        self.dec_feature_init = nn.Linear(latent_dim + bottleneck, feature_dim)

        if convlayer == 'gat':
            self.convlayer = torch_geometric.nn.GATConv
        elif convlayer == 'gcn':
            self.convlayer = torch_geometric.nn.GCNConv
        elif convlayer == 'gin':
            self.convlayer = lambda in_channels, out_channels: torch_geometric.nn.GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            )

        self.global_pool = torch_geometric.nn.global_max_pool

        if convlayer == 'gat':
            self.d_conv1 = self.convlayer(feature_dim + self.N, feature_dim, heads=n_head)
            self.layer_stack = nn.ModuleList([
                self.convlayer(feature_dim * n_head, feature_dim, heads=n_head)
                for _ in range(T - 1)
            ])
        else:
            self.d_conv1 = self.convlayer(feature_dim + self.N, feature_dim)
            self.layer_stack = nn.ModuleList([
                self.convlayer(feature_dim, feature_dim)
                for _ in range(T - 1)
            ])
            n_head = 1

        self.dec_pos = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_pos = nn.Linear(feature_dim, 2)

        self.dec_size = nn.Linear(feature_dim * n_head, feature_dim)
        self.fc_size = nn.Linear(feature_dim, 2)

        # self.dec_shape = nn.Linear(feature_dim * n_head, feature_dim)
        # self.fc_shape = nn.Linear(feature_dim, 6)
        #
        # self.dec_iou = nn.Linear(feature_dim * n_head, feature_dim)
        # self.fc_iou = nn.Linear(feature_dim, 1)

        # self.dec_exist = nn.Linear(feature_dim * n_head, feature_dim)
        # self.fc_exist = nn.Linear(feature_dim, 1)
        # self.fc_exist = nn.Linear(feature_dim * n_head, 1)

    def forward(self, z, condition, edge_index, batch):
        z = torch.cat([z, condition], dim=1)
        z = self.dec_feature_init(z)
        z = z[batch]

        one_hot = self.node_order_within_batch(batch)
        z = torch.cat([z, one_hot], 1)

        d_embed_0 = F.relu(z)
        d_embed_t = F.relu(self.d_conv1(d_embed_0, edge_index))
        for d_conv_t in self.layer_stack:
            d_embed_t = F.relu(d_conv_t(d_embed_t, edge_index))

        output_pos = F.relu(self.dec_pos(d_embed_t))
        output_pos = self.fc_pos(output_pos)

        output_size = F.relu(self.dec_size(d_embed_t))
        output_size = self.fc_size(output_size)

        # output_shape = F.relu(self.dec_shape(d_embed_t))
        # output_shape = F.softmax(self.fc_shape(output_shape), dim=-1)
        #
        # output_iou = F.relu(self.dec_iou(d_embed_t))
        # output_iou = F.sigmoid(self.fc_iou(output_iou))

        # output_exist = F.relu(self.dec_exist(d_embed_t))
        # output_exist = F.sigmoid(self.fc_exist(output_exist))
        # output_exist = F.sigmoid(self.fc_exist(d_embed_t))

        return output_pos, output_size

    def node_order_within_batch(self, batch):
        order_within_batch = torch.zeros_like(batch)
        unique_batches = batch.unique()
        for ub in unique_batches:
            mask = (batch == ub)
            order_within_batch[mask] = torch.arange(mask.sum(), device=batch.device)

        one_hot_order = torch.nn.functional.one_hot(order_within_batch, num_classes=120)
        return one_hot_order
class GraphCVAE(nn.Module):
    def __init__(self, T=3, feature_dim=256, latent_dim=256, n_head=8,
                 image_size=64, inner_channel=80, bottleneck=128,
                 condition_type='graph', convlayer='gat', batch_size=32):
        super(GraphCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.condition_type = condition_type
        self.batch_size = batch_size

        if condition_type == 'image':
            self.condition_encoder = BoundaryMaskEncoder(image_size=image_size, inner_channel=inner_channel,
                                                         bottleneck=bottleneck)
        elif condition_type == 'image_resnet34':
            self.condition_encoder = ResNet34(bottleneck=bottleneck)
        elif condition_type == 'graph':
            self.condition_encoder = GraphConditionEncoder(T=T, feature_dim=feature_dim, bottleneck=bottleneck,
                                                           n_head=n_head, convlayer=convlayer)

        self.encoder = GraphEncoder(T=T, feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head,
                                    convlayer=convlayer, batch_size=batch_size)
        self.decoder = GraphDecoder(T=T, feature_dim=feature_dim, latent_dim=latent_dim, n_head=n_head,
                                    bottleneck=bottleneck, convlayer=convlayer, batch_size=batch_size)

    def reparameterize(self, mu, logvar):
        return (torch.exp(0.5 * logvar)) * (torch.randn_like(torch.exp(0.5 * logvar))) + mu

    def forward(self, data):
        edge_index = data.edge_index
        mu, log_var = self.encoder(data, edge_index)
        z = self.reparameterize(mu, log_var)

        if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
            condition = self.condition_encoder(data.condition)
        else:
            condition = Batch.from_data_list(data.condition)
            condition = self.condition_encoder(condition, condition.edge_index)

        output_pos, output_size = self.decoder(z, condition, edge_index, data.batch)

        return output_pos, output_size, mu, log_var

    def test(self, data):
        z = torch.normal(mean=0, std=1, size=(1, self.latent_dim)).to(device=data.edge_index.device)

        if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
            condition = self.condition_encoder(data.condition)
        else:
            condition = Batch.from_data_list(data.condition)
            condition = self.condition_encoder(condition, condition.edge_index)

        output_pos, output_size = self.decoder(z, condition, data.edge_index, data.batch)

        return output_pos, output_size