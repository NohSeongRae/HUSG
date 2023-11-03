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
                      n_street=60, n_building=120, n_boundary=200, d_unit=8, d_street=64):
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
        'street_unit_position_datasets',
        'adj_matrices',
        'node_features'
    ]

    graphs = []
    a = []
    b = []
    c = []
    d = []
    e = []

    for city_name in tqdm(city_names):
        filepath = dataset_path + '/' + city_name + '/' + dataset_names[0] + '.pkl'
        with open(filepath, 'rb') as f:
            street_unit_position_datasets = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[1] + '.pkl'
        with open(filepath, 'rb') as f:
            adj_matrices = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[2] + '.pkl'
        with open(filepath, 'rb') as f:
            node_features = pickle.load(f)

        for idx in range(len(street_unit_position_datasets)):
            graph = nx.DiGraph(adj_matrices[idx])

            zeros = np.zeros((graph.number_of_nodes(), d_street, 2))
            street_feature = np.unique(street_unit_position_datasets[idx], axis=0)
            zeros[:len(street_feature)] = street_feature

            for node in graph.nodes():
                graph.nodes[node]['street_feature'] = zeros[node]

            zeros = np.zeros((graph.number_of_nodes(), 5))
            for i in range(len(node_features[idx])):
                if node_features[idx][i, 0] == 1:
                    node_features[idx][i, 5] = (node_features[idx][i, 5] + 1) / 2
                a.append(node_features[idx][i, 1])
                b.append(node_features[idx][i, 2])
                c.append(node_features[idx][i, 3])
                d.append(node_features[idx][i, 4])
                e.append(node_features[idx][i, 5])
            building_feature = node_features[idx][:, 1:]
            zeros[:len(building_feature)] = building_feature

            for node in graph.nodes():
                graph.nodes[node]['building_feature'] = zeros[node]

            zeros = np.zeros((graph.number_of_nodes(), 1))
            zeros[:len(street_feature)] = 1

            for node in graph.nodes():
                graph.nodes[node]['street_masks'] = zeros[node]

            zeros = np.zeros((graph.number_of_nodes(), 1))
            building_masks = node_features[idx][:, :1]
            zeros[:len(building_masks)] = building_masks

            for node in graph.nodes():
                graph.nodes[node]['building_masks'] = zeros[node]

            graphs.append(graph)

    print(np.array(sorted(a)))
    print(np.array(sorted(b)))
    print(np.array(sorted(c)))
    print(np.array(sorted(d)))
    print(np.array(sorted(e)))

    random.shuffle(graphs)

    total_size = len(graphs)

    for data_type in ['train', 'val', 'test']:
        if data_type == 'train':
            start_index = 0
            end_index = int(total_size * train_ratio)
        elif data_type == 'val':
            start_index = int(total_size * train_ratio)
            end_index = int(total_size * (train_ratio + val_ratio))
        else:
            start_index = int(total_size * (train_ratio + val_ratio))
            end_index = int(total_size * (train_ratio + val_ratio + test_ratio))

        save_path = './network/cvae_graph/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + data_type + '_datasets.gpickle', 'wb') as f:
            nx.write_gpickle(graphs[start_index:end_index], f)

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

    opt = parser.parse_args()

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
                         n_street=opt.n_street, n_building=opt.n_building,
                         n_boundary=opt.n_boundary, d_unit=opt.d_unit, d_street=opt.d_street)