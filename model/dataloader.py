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

def create_cycle_adjacency_matrix(num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adj[i, (i+1)%num_nodes] = 1
        adj[(i+1)%num_nodes, i] = 1
    return adj

class FloorDataset(Dataset):
    def __init__(self, npy_folder_path, scaler):
        self.npy_files = glob.glob(f"{npy_folder_path}/*.npy")
        self.adjacency_matrix = create_cycle_adjacency_matrix(596)
        self.scaler = scaler

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_file = self.npy_files[idx]
        data = np.load(npy_file)

        zero_indices = data == 0
        non_zero_data = data[~zero_indices]

        # Transform the data
        scaled_non_zero_data = self.scaler.transform(non_zero_data.reshape(-1, 1)).flatten()

        transformed_data = np.zeros_like(data)
        transformed_data[~zero_indices] = scaled_non_zero_data

        features = torch.tensor(transformed_data).float().view(596, 9)

        return features, self.adjacency_matrix

