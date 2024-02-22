import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
import random
from tqdm import tqdm

from model import GraphCVAE
from dataloader import GraphDataset
from visualization import plot

def test(d_feature, d_latent, n_head, T, checkpoint_epoch, save_dir_path, condition_type, convlayer):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = GraphDataset(data_type='test', condition_type=condition_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    cvae = GraphCVAE(T=T, feature_dim=d_feature, latent_dim=d_latent, n_head=n_head,
                     condition_type=condition_type, convlayer=convlayer).to(device=device)

    if checkpoint_epoch == 0:
        checkpoint_epoch = 'best'
    checkpoint = torch.load("./models/" + save_dir_path + "/epoch_" + str(checkpoint_epoch) + ".pth")
    cvae.load_state_dict(checkpoint['model_state_dict'])

    cvae.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            data, polygon_path, data_path  = data

            data = data.to(device=device)
            output_pos, output_size, output_theta, output_exist = cvae.test(data)

            if condition_type == 'image' or condition_type == 'image_resnet34':
                plot(output_pos.detach().cpu().numpy(),
                     output_size.detach().cpu().numpy(),
                     output_theta.detach().cpu().numpy(),
                     output_exist.detach().cpu().numpy(),
                     data.exist_features.detach().cpu().numpy(),
                     data.node_features.detach().cpu().numpy(),
                     idx + 1,
                     condition_type,
                     polygon_path,
                     save_dir_path,
                     data_path)

            elif condition_type == 'graph':
                plot(output_pos.detach().cpu().numpy(),
                     output_size.detach().cpu().numpy(),
                     output_theta.detach().cpu().numpy(),
                     data.building_mask.detach().cpu().numpy(),
                     data.node_features.detach().cpu().numpy(),
                     idx + 1,
                     condition_type,
                     polygon_path,
                     save_dir_path,
                     data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--T", type=int, default=3, help="Dimension of the model.")
    parser.add_argument("--d_feature", type=int, default=256, help="Dimension of the model.")
    parser.add_argument("--d_latent", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_head", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--save_dir_path", type=str, default="cvae_graph", help="save dir path")
    parser.add_argument("--condition_type", type=str, default='image_resnet34', help="save dir path")
    parser.add_argument("--convlayer", type=str, default='gat', help="save dir path")

    opt = parser.parse_args()

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    test(d_feature=opt.d_feature, d_latent=opt.d_latent, n_head=opt.n_head, T=opt.T,
         checkpoint_epoch=opt.checkpoint_epoch, save_dir_path=opt.save_dir_path,
         condition_type=opt.condition_type, convlayer=opt.convlayer)