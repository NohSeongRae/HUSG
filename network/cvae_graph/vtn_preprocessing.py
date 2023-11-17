import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import networkx as nx
import random
from torch_geometric.utils import dense_to_sparse, to_dense_adj


def preprocesing_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                         n_street=60, n_building=120, n_boundary=200, d_unit=8, d_street=64, condition_type='graph'):
    dataset_path = '../../datasets/HUSG/'
    # dataset_path = '../../datasets/HUSG/'
    # dataset_path = '../mnt/2_transformer/train_dataset'
    # dataset_path = 'Z:/iiixr-drive/Projects/2023_City_Team/2_transformer/train_dataset'
    # dataset_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'train_dataset')

    city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
                  "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
                  "sanfrancisco", "miami", "seattle", "boston", "providence",
                  "neworleans", "denver", "pittsburgh", "washington"]

    dataset_names = [
        'node_features',
        'building_semantics',
        'boundary_filenames',
        'insidemask_filename',
    ]

    for city_name in tqdm(city_names):
        filepath = dataset_path + '/' + city_name + '/' + dataset_names[0] + '.pkl'
        with open(filepath, 'rb') as f:
            node_features = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[1] + '.pkl'
        with open(filepath, 'rb') as f:
            building_semantics = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[2] + '.pkl'
        with open(filepath, 'rb') as f:
            source_file_names = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[3] + '.pkl'
        with open(filepath, 'rb') as f:
            mask_file_names = pickle.load(f)

        for idx in range(len(node_features)):
            n_node = len(node_features[idx])
            n_building = len(building_semantics[idx])
            n_chunk = n_node - n_building

            if np.any((node_features[idx][:, :2] < -0.1) | (node_features[idx][:, :2] > 1.1)):
                print(source_file_names[idx], idx, node_features[idx][:, :2])
                continue

            node_feature = []
            for i in range(len(node_features[idx])):
                if i < n_chunk:
                    node_feature.append([0] + node_features[idx].tolist())
                else:
                    node_feature.append([1] + node_features[idx].tolist())

            save_path = './network/cvae_graph/vtn_train_datasets/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with open(save_path + '/' + mask_file_names[idx] + '.pkl', 'wb') as f:
                pickle.dump(node_feature, f)


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--d_street", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--d_unit", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--n_building", type=int, default=120, help="binary classification for building existence.")
    parser.add_argument("--n_boundary", type=int, default=250, help="Number of boundary or token.")
    parser.add_argument("--n_street", type=int, default=60, help="Number of boundary or token.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--condition_type", type=str, default="graph",
                        help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
                         n_street=opt.n_street, n_building=opt.n_building,
                         n_boundary=opt.n_boundary, d_unit=opt.d_unit, d_street=opt.d_street,
                         condition_type=opt.condition_type)
