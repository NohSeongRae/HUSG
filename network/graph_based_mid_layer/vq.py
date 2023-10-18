import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, d_embed, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.n_embed = n_embed
        self.d_embed = d_embed

        self.embed = nn.Embedding(n_embed, d_embed)
        self.embed.weight.data.uniform_(-1/self.n_embed, 1/self.n_embed)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape

        x_flat = x.view(-1, self.d_embed)

        distances = (torch.sum(x_flat ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embed.weight ** 2, dim=1)
                     - 2 * torch.matmul(x_flat, self.embed.weight.t()))

        enc_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        enc = torch.zeros(enc_indices.shape[0], self.n_embed, device=x.device)
        enc.scatter_(1, enc_indices, 1)

        quantized = torch.matmul(enc, self.embed.weight).view(x_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return quantized.permute(0, 3, 1, 2).contiguous(), loss