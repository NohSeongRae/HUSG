# train.py
from torch.optim.lr_scheduler import StepLR
from floorVAE import VAE
from dataloader import FloorDataset
import torch
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import joblib

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_features, batch_adj in loader:
        batch_features, batch_adj = batch_features.to(device), batch_adj.to(device)

        optimizer.zero_grad()

        reconstructed_features, mu, logvar = model(batch_features, batch_adj)

        reconstruction_loss = mse_loss(reconstructed_features, batch_features)

        epsilon = 1e-8
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar.exp() + epsilon))

        loss = reconstruction_loss + kl_divergence
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == '__main__':
    dataset_path = 'Z:/iiixr-drive/Projects/2023_City_Team/features'

    scaler = joblib.load('scaler.pkl')
    dataset = FloorDataset(dataset_path, scaler)

    # 데이터셋을 학습용과 검증용으로 분리
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    feature_dim = 9
    latent_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    floorNet = VAE(feature_dim, latent_dim).to(device)
    optimizer = optim.Adam(floorNet.parameters(), lr=0.005, weight_decay=0.001)

    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(f'runs/floorNet_experiment/{current_time}')

    num_epochs = 500
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(floorNet, train_loader, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)

        scheduler.step()

    # 모델 저장
    torch.save(floorNet.state_dict(), f'floorNet_{num_epochs}_{current_time}.pth')
