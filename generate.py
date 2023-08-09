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
adj_dim = 596
latent_dim = 256

# Load the model
loaded_floorNet = VAE(feature_dim, latent_dim)
loaded_floorNet.load_state_dict(torch.load('floorNet.pth'))
loaded_floorNet.eval()

# Generate new data
def generate_data(model, num_samples, adj_matrix):
    with torch.no_grad():
        latent_samples = torch.randn(num_samples, latent_dim)
        expanded_latent = latent_samples.unsqueeze(1).expand(-1, adj_matrix.size(0), -1)

        adj_matrix_batch = adj_matrix.unsqueeze(0).repeat(num_samples, 1, 1)  # [num_samples, adj_dim, adj_dim]
        generated_data = model.decoder(expanded_latent, adj_matrix_batch)
    return generated_data


adj_matrix = torch.eye(adj_dim)
num_samples_to_generate = 10
new_data = generate_data(loaded_floorNet, num_samples_to_generate, adj_matrix)

print("output: ", new_data.shape)
