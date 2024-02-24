import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch

class GATEncoder(nn.Module):
    def __init__(self, T, feature_dim, latent_dim, n_head):
        super(GATEncoder, self).__init__()
        #self.positional_encoding= 진짜 필요할까? 좀 더 생각해보자
        self.mask_embed = nn.Embedding(2, feature_dim)
        self.node_fc = nn.Linear(feature_dim + feature_dim, feature_dim)
        self.conv_layer=torch_geometric.nn.GATConv
        self.global_pool=torch_geometric.nn.global_max_pool

        self.conv=self.conv_layer(feature_dim, feature_dim)
        self.layer_stack=nn.ModuleList(
            [
                self.conv_layer(feature_dim, feature_dim)
                for _ in range(T -1)
            ]
        )
        self.aggregate = nn.Linear(int(feature_dim * (1.0 +T)), bottleneck)
    def forward(self, data, edge_index):
        street_feature = data.condition_street_feature
        street_feature = self.bbox_fc(street_feature)
        street_feature = F.relu(street_feature)

        node_embed_init=street_feature
        graph_embed_init=self.global_pool(node_embed_init, data.batch)

        node_embed=F.relu(self.conv(node_embed_init, edge_index))
        graph_embed=self.global_pool(node_embed, data.batch)

        graph_embed_stack=torch.cat((graph_embed_init, graph_embed), dim=1)

        for conv in self.layer_stack:
            node_embed = F.relu(conv(node_embed, edge_index))
            graph_embed=self.global_pool(node_embed, data.batch)
            graph_embed_stack=torch.cat((graph_embed_stack, graph_embed), dim=1)

        graph_latent_vector=self.aggregate(graph_embed_stack)

        return graph_latent_vector



class GATConditionalEncoder(nn.Module):
    def __init__(self, T, feature_dim, bottleneck, n_head):
        super(GATEncoder).__init__()
        # self.positional_encoding= 진짜 필요할까? 좀 더 생각해보자
        self.conv_layer = torch_geometric.nn.GATConv
        self.global_pool = torch_geometric.nn.global_max_pool

        self.conv = self.conv_layer(feature_dim, feature_dim)
        self.layer_stack = nn.ModuleList(
            [
                self.conv_layer(feature_dim * n_head, feature_dim, heads=n_head)
                for _ in range(T - 1)
            ]
        )
        self.aggregate = nn.Linear(int(feature_dim * (1.0 + n_head * T)), bottleneck)

    def forward(self, data, edge_index):
        street_feature = data.condition_street_feature
        street_feature = self.bbox_fc(street_feature)
        street_feature = F.relu(street_feature)

        node_embed_init = street_feature
        graph_embed_init = self.global_pool(node_embed_init, data.batch)

        node_embed = F.relu(self.conv(node_embed_init, edge_index))
        graph_embed = self.global_pool(node_embed, data.batch)

        graph_embed_stack = torch.cat((graph_embed_init, graph_embed), dim=1)

        for conv in self.layer_stack:
            node_embed = F.relu(conv(node_embed, edge_index))
            graph_embed = self.global_pool(node_embed, data.batch)
            graph_embed_stack = torch.cat((graph_embed_stack, graph_embed), dim=1)

        graph_latent_vector = self.aggregate(graph_embed_stack)

        return graph_latent_vector
class TransformerDecoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
class GraphConditionalVQVAE(nn.Module):
    def __init__(self):
        pass
    def reparameterize(self):
        pass
    def forward(self, data):

    def test(self, data):
        pass