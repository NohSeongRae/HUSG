import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import pandas as pd

city_name = 'milan'

# 학습 코드

# cycle graph 생성
def create_cycle_adjacency_matrix(num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adj[i, (i+1)%num_nodes] = 1  # Connect to next node
        adj[(i+1)%num_nodes, i] = 1  # Connect to previous node (for undirected graph)
    return adj


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionalLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        h = torch.bmm(adj, x)
        batch_size, num_nodes, feature_dim = h.size()
        h = h.view(batch_size * num_nodes, feature_dim)
        h = self.fc(h)
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
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.gc = GraphConvolutionalLayer(256, feature_dim)

    def forward(self, x, adj):
        x = self.mlp(x)
        return self.gc(x, adj)

class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = FloorNetEncoder(feature_dim, latent_dim)
        self.decoder = FloorNetDecoder(latent_dim, feature_dim)

    def forward(self, x, adj):
        mu, logvar = self.encoder(x, adj)
        latent_vector = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon = self.decoder(latent_vector, adj)
        return recon, mu, logvar

# dataset
class FloorDataset(Dataset):
    def __init__(self, csv_folder_path):
        self.csv_files = glob.glob(f"{csv_folder_path}/*.csv")
        self.adjacency_matrix = create_cycle_adjacency_matrix(592)  # 592x592 matrix

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        data = pd.read_csv(csv_file, header=None)

        # NaN 값들을 0으로 채우기
        data.fillna(0, inplace=True)

        features = torch.tensor(data.values).float().view(592, 9)  # 592x9 matrix
        return features, self.adjacency_matrix


# Dataset Initialization
feature_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'result')

csv_folder_path = feature_path
adj_dim = 592
dataset = FloorDataset(csv_folder_path)

# Dataset and DataLoader Initialization
feature_dim = 9

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model and Optimizer Initialization
latent_dim = 256
floorNet = VAE(feature_dim, latent_dim)
optimizer = optim.Adam(floorNet.parameters(), lr=0.001)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_features, batch_adj in dataloader:
        optimizer.zero_grad()

        # print("batch_features shape:", batch_features.shape)
        # print("batch_adj shape:", batch_adj.shape)

        reconstructed_features, mu, logvar = floorNet(batch_features, batch_adj)

        reconstruction_loss = nn.MSELoss()(reconstructed_features, batch_features)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = reconstruction_loss + kl_divergence
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(floorNet.state_dict(), 'floorNet.pth')