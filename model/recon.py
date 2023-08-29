import torch
from floorVAE import VAE
from dataloader import FloorDataset
from torch.utils.data import DataLoader
import numpy as np
import joblib

def create_cycle_adjacency_matrix(num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adj[i, (i+1) % num_nodes] = 1
        adj[(i+1) % num_nodes, i] = 1
    return adj

def restore_data(data, scaler):
    restored_data = scaler.inverse_transform(data)
    return restored_data

def test_model(model, loader, device):
    model.eval()
    reconstructed_features_list = []
    original_features_list = []
    with torch.no_grad():
        for batch_features, batch_adj in loader:
            batch_features, batch_adj = batch_features.to(device), batch_adj.to(device)
            reconstructed_features, _, _ = model(batch_features, batch_adj)
            reconstructed_features_list.append(reconstructed_features.cpu())
            original_features_list.append(batch_features.cpu())

    original_features_array = np.vstack([tensor.numpy() for tensor in original_features_list])
    reconstructed_features_array = np.vstack([tensor.numpy() for tensor in reconstructed_features_list])

    return original_features_array, reconstructed_features_array

if __name__ == '__main__':
    scaler = joblib.load('scaler.pkl')

    dataset_path = 'Z:/iiixr-drive/Projects/2023_City_Team/features'  # 데이터셋 경로 설정
    dataset = FloorDataset(dataset_path, scaler)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    feature_dim = 9
    latent_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    floorNet = VAE(feature_dim, latent_dim).to(device)

    floorNet.load_state_dict(torch.load('trained_model/floorNet_500.pth'))

    reconstructed_features_array, original_features_array = test_model(floorNet, test_loader, device)

    # Restore the data to its original scale
    restored_reconstructed_features_array = restore_data(reconstructed_features_array, scaler)
    restored_original_features_array = restore_data(original_features_array, scaler)

    print(restored_reconstructed_features_array[1])
    print(restored_original_features_array[1])

    np.save('restored_recon2.npy', restored_reconstructed_features_array[1])
    np.save('restored_original2.npy', restored_original_features_array[1])
