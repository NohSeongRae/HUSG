import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib


# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 학습 코드

# cycle graph 생성
def create_cycle_adjacency_matrix(num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adj[i, (i+1)%num_nodes] = 1
        adj[(i+1)%num_nodes, i] = 1
    return adj


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionalLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, adj):
        h = torch.bmm(adj, x)
        batch_size, num_nodes, feature_dim = h.size()
        h = h.view(batch_size * num_nodes, feature_dim)
        h = self.fc(h)
        h = self.bn(h)  # 배치 정규화 적용
        h = h.view(batch_size, num_nodes, self.output_dim)
        return F.relu(h)



class FloorNetEncoder(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(FloorNetEncoder, self).__init__()
        self.gc1 = GraphConvolutionalLayer(feature_dim, 128)
        self.gc2 = GraphConvolutionalLayer(128, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class FloorNetDecoder(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super(FloorNetDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 596),
            nn.BatchNorm1d(596),
            nn.ReLU(),
            nn.Linear(596, 256),
            nn.ReLU()
        )

        self.gc1 = GraphConvolutionalLayer(256, 128)
        self.gc2 = GraphConvolutionalLayer(128, feature_dim)

    def forward(self, x, adj):
        batch_size = x.size(0)
        x = self.mlp(x)  # MLP
        x = x.view(batch_size, 596, 256)

        x = self.gc1(x, adj)  # GCN
        return self.gc2(x, adj)  # 두 GCN



class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = FloorNetEncoder(feature_dim, latent_dim)
        self.decoder = FloorNetDecoder(latent_dim, feature_dim)

    def forward(self, x, adj):
        # print("x.shape: ", x.shape)
        # print("adj.shape: ", adj.shape)
        mu, logvar = self.encoder(x, adj)
        latent_vector = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon = self.decoder(latent_vector, adj)
        return recon, mu, logvar

# dataset
