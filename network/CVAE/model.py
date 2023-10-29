import torch
from torch import nn
from torch_geometric.nn import GCNConv

class CVAE(nn.Module):
    def __init__(self, parcel_dim, bbox_dim, hidden_dim=50, latent_dim=20):
        super(CVAE, self).__init__()

        self.condition_encoder = nn.Sequential(
            nn.Linear(bbox_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.encoder = nn.Sequential(
            GCNConv(parcel_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, 2 * latent_dim),  # for mean and variance
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            GCNConv(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, parcel_dim),
            nn.ReLU(),
            nn.Linear(parcel_dim, parcel_dim),
            nn.ReLU(),
            nn.Linear(parcel_dim, parcel_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, bbox, parcel=None):
        c = self.condition_encoder(bbox)

        if parcel is not None:
            # Encode
            h = self.encoder(torch.cat([parcel, c], dim=1))
            mu, logvar = torch.chunk(h, 2, dim=1)

            # Reparameterize
            z = self.reparameterize(mu, logvar)
        else:
            # Generate random z for inference
            z = torch.randn((bbox.size(0), self.encoder[2].out_features // 2)).to(bbox.device)

        # Decode
        return self.decoder(torch.cat([z, c], dim=1))
