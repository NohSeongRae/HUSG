import os
import argparse
import torch
from torch.utils.data import DataLoader

import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from model import GraphTransformer
from dataloader import GraphDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="transformer_graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="save dir path")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    device = torch.device('cpu')

    test_dataset = GraphDataset(data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    transformer = GraphTransformer(opt.d_model, opt.d_model * 4, opt.n_layer, opt.n_head, opt.dropout).to(device=device)

    checkpoint = torch.load("./models/" + opt.save_dir_path + "/epoch_"+ 'best' + ".pth")
    transformer.load_state_dict(checkpoint['model_state_dict'])

    transformer.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            data, file = data
            file = file[0].replace('.pickle', '')

            building_adj_matrix_padded = data['building_adj_matrix_padded'].to(device=device)
            boundary_adj_matrix_padded = data['boundary_adj_matrix_padded'].to(device=device)
            bb_adj_matrix_padded = data['bb_adj_matrix_padded'].to(device=device)
            boundary_pos_padded = data['boundary_pos_padded'].to(device=device)
            building_pad_mask = data['building_pad_mask'].to(device=device)
            boundary_pad_mask = data['boundary_pad_mask'].to(device=device)
            bb_pad_mask = data['bb_pad_mask'].to(device=device)
            n_boundary = data['n_boundary']
            n_building = data['n_building']

            output = transformer(building_adj_matrix_padded, boundary_adj_matrix_padded,
                                 building_pad_mask, boundary_pad_mask, boundary_pos_padded)
            output *= bb_pad_mask
            output = (output >= 0.5).float()

            pred_adj_matrix = np.zeros((n_boundary + n_building, n_boundary + n_building))
            gt_adj_matrix = np.zeros((n_boundary + n_building, n_boundary + n_building))

            pred_adj_matrix[:n_boundary, :n_boundary] = boundary_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_boundary, :n_boundary]
            pred_adj_matrix[n_boundary:, n_boundary:] = building_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_building, :n_building]
            pred_adj_matrix[n_boundary:, :n_boundary] = output.squeeze(0).detach().cpu().numpy()[:n_building, :n_boundary]
            pred_adj_matrix[:n_boundary, n_boundary:] = output.squeeze(0).detach().cpu().numpy().T[:n_boundary, :n_building]

            gt_adj_matrix[:n_boundary, :n_boundary] = boundary_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_boundary, :n_boundary]
            gt_adj_matrix[n_boundary:, n_boundary:] = building_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_building, :n_building]
            gt_adj_matrix[n_boundary:, :n_boundary] = bb_adj_matrix_padded.squeeze(0).detach().cpu().numpy()[:n_building, :n_boundary]

            path = f'../preprocessing/global_mapper/ours_graph_datasets/test/{file}.gpickle'
            graph = nx.read_gpickle(path)

            graph.remove_edges_from(list(graph.edges()))

            for i, row in enumerate(pred_adj_matrix):
                for j, val in enumerate(row):
                    if val == 1:
                        graph.add_edge(i, j)

            path = path.replace('ours_graph_datasets', f'synthetic_datasets')

            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            nx.write_gpickle(graph, path)