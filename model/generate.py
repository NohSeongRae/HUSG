import torch
from floorVAE import VAE, create_cycle_adjacency_matrix
import numpy as np

feature_dim = 9
latent_dim = 256
num_nodes = 596
batch_size = 1
model = VAE(feature_dim, latent_dim)

model.load_state_dict(torch.load('floorNet_500.pth'))

model.eval()

with torch.no_grad():
    z = torch.randn(batch_size, num_nodes, latent_dim)
    adj = create_cycle_adjacency_matrix(num_nodes)
    adj = adj.unsqueeze(0)
    new_data = model.decoder(z, adj)

    # 생성된 데이터를 numpy 배열로 변환
    new_data_numpy = new_data.cpu().numpy()

    # numpy 배열을 .npy 파일로 저장
    np.save('new_data.npy', new_data_numpy)

    print(new_data)
