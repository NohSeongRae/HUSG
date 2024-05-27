import argparse
import torch
from torch_geometric.loader import DataLoader

import numpy as np
import random
from tqdm import tqdm

from model import GraphCVAE
from dataloader import GraphDataset
from visualization import plot

def test(d_feature, d_latent, n_head, T, checkpoint_epoch, save_dir_path, condition_type, convlayer):
    """
    Tests the model and visualizes the results.

    Parameters:
    - d_feature (int): Dimension of the features in the model.
    - d_latent (int): Dimension of the latent vectors.
    - n_head (int): Number of attention heads.
    - T (int): The number of layers or time steps in the model.
    - checkpoint_epoch (int/str): Epoch number of the checkpoint to load for testing. Use 'best' to load the best checkpoint.
    - save_dir_path (str): Directory path where the model checkpoints are saved.
    - condition_type (str): The type of condition used in the model (e.g., 'image', 'image_resnet34', 'graph').
    - convlayer (str): The type of convolutional layer used in the model (e.g., 'gat', 'gcn', 'gin').

    The function loads the test dataset based on the specified condition type, initializes the GraphCVAE model with the provided
    parameters, loads the model weights from the specified checkpoint, and evaluates the model on the test dataset.
    The results are visualized using the 'plot' function, which is expected to handle different condition types and
    generate appropriate visualizations.
    """

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
        count = 1
        for idx, data in enumerate(tqdm(test_dataloader)):
            data, polygon_path, data_path  = data

            data = data.to(device=device)
            output_pos, output_size, output_theta = cvae.test(data)

            if condition_type == 'image' or condition_type == 'image_resnet34':
                plot(output_pos.detach().cpu().numpy(),
                     output_size.detach().cpu().numpy(),
                     output_theta.detach().cpu().numpy(),
                     data.building_mask.detach().cpu().numpy(),
                     data.node_features.detach().cpu().numpy(),
                     idx + 1,
                     condition_type,
                     polygon_path,
                     save_dir_path,
                     data_path[0])

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
                     data_path[0])

            count += 1
            if count % 1001 == 0:
                return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initializes a GraphCVAE model with user-defined hyperparameters for testing.")

    parser.add_argument("--T", type=int, default=3, help="Number of transformation layers or depth of the GraphCVAE model.")
    parser.add_argument("--d_feature", type=int, default=256, help="Dimensionality of the input feature vectors in the GraphCVAE model.")
    parser.add_argument("--d_latent", type=int, default=512, help="Size of the latent space in the GraphCVAE model.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of heads in the multi-head attention mechanism of the GraphCVAE model.")
    parser.add_argument("--seed", type=int, default=327, help="Seed for random number generators to ensure reproducibility.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Epoch number of the model checkpoint to load for testing. Use 0 to specify the latest checkpoint.")
    parser.add_argument("--save_dir_path", type=str, default="cvae_graph", help="Directory path where the model checkpoints and test outputs are saved")
    parser.add_argument("--condition_type", type=str, default='image_resnet34', help="Type of conditional input used by the GraphCVAE model.")
    parser.add_argument("--convlayer", type=str, default='gat', help="Type of convolutional layer used in the GraphCVAE model (e.g., 'gat', 'gcn', 'gin').")

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