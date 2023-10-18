import torch.nn as nn
from network.graph_based_mid_layer.encoder import Encoder
from network.graph_based_mid_layer.vq import VectorQuantizer
from network.graph_based_mid_layer.decoder import Decoder

class VQVAE(nn.Module):
    def __init__(self, d_in, d_model, d_out, n_embed, n_res_layer, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(d_in=d_in, d_model=d_model, n_res_layer=n_res_layer)
        self.vq = VectorQuantizer(n_embed=n_embed, d_embed=d_model, commitment_cost=commitment_cost)
        self.decoder = Decoder(d_out=d_out, d_model=d_model, n_res_layer=n_res_layer)

    def forward(self, x):
        z = self.encoder(x)
        quantized, loss = self.vq(z)
        x_recon = self.decoder(quantized)

        return x_recon, loss