import torch
import torch.nn as nn
import torch.nn.functional as F

# 학습된 모델 불러와서 새로운 데이터 생성

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
        x = x.unsqueeze(1).repeat(1, adj.size(1), 1)  # This will change the shape to [num_samples, adj_dim, 256]
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

# Model Initialization
feature_dim = 9
adj_dim = 592
latent_dim = 256

# Load the model
loaded_floorNet = VAE(feature_dim, latent_dim)
loaded_floorNet.load_state_dict(torch.load('floorNet.pth'))
loaded_floorNet.eval()

# Generate new data
# 학습된 모델 불러오기
loaded_floorNet = VAE(feature_dim, latent_dim)
loaded_floorNet.load_state_dict(torch.load('floorNet.pth'))
loaded_floorNet.eval()

# Latent space의 통계 계산
def get_latent_statistics(model, data_loader):
    mus, logvars = [], []

    with torch.no_grad():
        for batch_features, batch_adj in data_loader:
            batch_features, batch_adj = batch_features.to(device), batch_adj.to(device)
            mu, logvar = model.encoder(batch_features, batch_adj)
            mus.append(mu)
            logvars.append(logvar)

    mus = torch.cat(mus, dim=0)
    logvars = torch.cat(logvars, dim=0)

    latent_mean = mus.mean(dim=0)
    latent_std = (logvars.exp()).mean(dim=0).sqrt()

    return latent_mean, latent_std

# 새로운 데이터 생성
def generate_data_using_statistics(model, num_samples, adj_matrix, latent_mean, latent_std):
    with torch.no_grad():
        latent_samples = torch.randn(num_samples, latent_dim) * latent_std + latent_mean
        adj_matrix_batch = adj_matrix.unsqueeze(0).repeat(num_samples, 1, 1)
        generated_data = model.decoder(latent_samples, adj_matrix_batch)

    return generated_data

if __name__ == '__main__':
    # Dataset Initialization
    feature_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'result')
    csv_folder_path = feature_path
    adj_dim = 592
    dataset = FloorDataset(csv_folder_path)

    # Dataset and DataLoader Initialization
    feature_dim = 9
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Model and Optimizer Initialization
    latent_dim = 256
    floorNet = VAE(feature_dim, latent_dim).to(device)
    optimizer = optim.Adam(floorNet.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_features, batch_adj in dataloader:
            batch_features, batch_adj = batch_features.to(device), batch_adj.to(device)
            optimizer.zero_grad()

            reconstructed_features, mu, logvar = floorNet(batch_features, batch_adj)

            reconstruction_loss = nn.MSELoss()(reconstructed_features, batch_features)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(floorNet.state_dict(), 'floorNet.pth')

    # Latent space의 통계 계산
    latent_mean, latent_std = get_latent_statistics(loaded_floorNet, dataloader)

    # 새로운 데이터 생성
    adj_matrix = create_cycle_adjacency_matrix(adj_dim)
    num_samples_to_generate = 20
    new_data = generate_data_using_statistics(loaded_floorNet, num_samples_to_generate, adj_matrix, latent_mean, latent_std)

    print("Generated data shape:", new_data.shape)
