import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

from torch.utils.tensorboard import SummaryWriter

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.gc1 = GraphConvolutionalLayer(256, 128)  # 첫 번째 GCN
        self.gc2 = GraphConvolutionalLayer(128, feature_dim)  # 두 번째 GCN

    def forward(self, x, adj):
        x = self.mlp(x)  # 다층 퍼셉트론 (MLP)
        x = self.gc1(x, adj)  # 첫 번째 GCN
        return self.gc2(x, adj)  # 두 번째 GCN


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

from sklearn.preprocessing import MinMaxScaler

# ...
class FloorDataset(Dataset):
    def __init__(self, csv_folder_path):
        self.csv_files = glob.glob(f"{csv_folder_path}/*.csv")
        self.adjacency_matrix = create_cycle_adjacency_matrix(592)  # 592x592 matrix
        self.scaler = MinMaxScaler()

        # 모든 CSV 파일에서 데이터 로드
        all_data = []
        for csv_file in self.csv_files:
            data = pd.read_csv(csv_file, header=None)
            all_data.append(data)

        all_data = pd.concat(all_data, axis=0)

        # NaN 값들을 0으로 채우기
        all_data.fillna(0, inplace=True)

        # 데이터 세트 전체에 대한 스케일러를 맞추기
        self.scaler.fit(all_data)

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        data = pd.read_csv(csv_file, header=None)

        # NaN 값들을 0으로 채우기
        data.fillna(0, inplace=True)

        # 0인 값을 가진 위치를 저장
        zero_indices = data == 0

        print(data)

        # 0이 아닌 값들만 스케일링
        non_zero_data = data[~zero_indices]
        scaled_non_zero_data = self.scaler.transform(non_zero_data.values.reshape(-1, 1)).flatten()

        print(non_zero_data.shape)
        print(scaled_non_zero_data.shape)

        if np.isnan(scaled_non_zero_data).any():
            raise ValueError("NaN value detected after scaling!")

        # 스케일링된 데이터를 원래 위치에 삽입
        non_zero_positions = zero_indices.stack()[zero_indices.stack() == False].index
        for idx, pos in enumerate(non_zero_positions):
            data.at[pos] = scaled_non_zero_data[idx]

        # 데이터를 텐서로 변환
        features = torch.tensor(data.values).float().view(592, 9)

        print(features)

        return features, self.adjacency_matrix


if __name__ == '__main__':
    writer = SummaryWriter('runs/floorNet_experiment')
    feature_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'result')
    npy_folder_path = feature_path
    dataset = FloorDataset(npy_folder_path)
    adj_dim = 592

    # Dataset and DataLoader Initialization
    feature_dim = 9
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Model and Optimizer Initialization
    latent_dim = 256
    floorNet = VAE(feature_dim, latent_dim).to(device)  # 모델을 GPU로 옮김
    optimizer = optim.Adam(floorNet.parameters(), lr=0.001, weight_decay=0.001)

    features, adj = next(iter(dataloader))
    writer.add_graph(floorNet, (features.to(device), adj.to(device)))

    # Training Loop
    num_epochs = 500
    for epoch in range(num_epochs):
        for batch_idx, (batch_features, batch_adj) in enumerate(dataloader):
            batch_features, batch_adj = batch_features.to(device), batch_adj.to(device)  # 데이터를 GPU로 옮김
            optimizer.zero_grad()

            reconstructed_features, mu, logvar = floorNet(batch_features, batch_adj)

            reconstruction_loss = nn.MSELoss()(reconstructed_features, batch_features)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    scaler_filename = 'scaler.pkl'
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(dataset.scaler, scaler_file)

    # Save the model
    torch.save(floorNet.state_dict(), 'floorNet_500.pth')

    writer.close()