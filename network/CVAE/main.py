import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


N=3
batch_size=1
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feat_mat=torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]).to(device)
size_mat=torch.randn_like(feat_mat)
one_hot = torch.eye(N, dtype=torch.float32).to(device).repeat(batch_size, 1)
print(f'feat_mat: {feat_mat}\n feat_mat.shape: {feat_mat.shape}\n one_hot: {one_hot}\none_hot.shape: {one_hot.shape}')

x=torch.cat([feat_mat,one_hot], dim=1)
print(f'x: {x}\n x.shape: {x.shape}')

print(f'size_mat: {size_mat}\n size_mat.shape: {size_mat.shape}')
x=torch.cat([size_mat, x], dim=1)
print('size_mat + x')
print(f'x: {x}\n x.shape: {x.shape}')


