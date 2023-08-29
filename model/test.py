# test.py
import torch
from floorVAE import VAE, FloorDataset
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss


def test_model(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_features, batch_adj in loader:
            batch_features, batch_adj = batch_features.to(device), batch_adj.to(device)
            reconstructed_features, mu, logvar = model(batch_features, batch_adj)
            reconstruction_loss = mse_loss(reconstructed_features, batch_features)
            total_loss += reconstruction_loss.item()
    return total_loss / len(loader)


if __name__ == '__main__':
    dataset_path = 'Z:/iiixr-drive/Projects/2023_City_Team/features'  # 데이터셋 경로 설정
    dataset = FloorDataset(dataset_path)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    feature_dim = 9
    latent_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    floorNet = VAE(feature_dim, latent_dim).to(device)

    # 모델 불러오기
    floorNet.load_state_dict(torch.load('floorNet_500.pth'))

    test_loss = test_model(floorNet, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}')
