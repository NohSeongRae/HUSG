import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import networkx as nx
import random

def preprocesing_dataset(condition_type='graph'):

    dataset_path = f'C:/Users/Dobby/Downloads/processed_block_building_info.pkl'
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    dataset_names = [
        'edge_indices',
        'node_features',
        'boundary_mask',
        'building_polygons'
    ]

    for file_idx in range(len(data)):
        edge_indices = data[file_idx][dataset_names[0]]
        node_features = data[file_idx][dataset_names[1]]
        boundary_mask = data[file_idx][dataset_names[2]]
        building_polygons = data[file_idx][dataset_names[3]]

        graph = nx.Graph()
        graph.add_edges_from(edge_indices)
        adj_matrix = nx.adjacency_matrix(graph).todense()

        n_node = graph.number_of_nodes()
        n_building = len(building_polygons)
        n_chunk = n_node - n_building

        if condition_type == 'graph':
            street_graph = adj_matrix[:n_chunk, :n_chunk]
            street_graph = nx.DiGraph(street_graph)

            chunk_feature = node_features[:n_chunk]
            for node in street_graph.nodes():
                street_graph.nodes[node]['chunk_features'] = chunk_feature[node]

            graph.graph['condition'] = street_graph
        elif condition_type == 'image':
            graph.graph['condition'] = boundary_mask

        for node in graph.nodes():
            graph.nodes[node]['node_features'] = node_features[node]

        zeros = np.zeros((graph.number_of_nodes(), 1))
        zeros[n_chunk:] = 1

        for node in graph.nodes():
            graph.nodes[node]['building_masks'] = zeros[node]

        save_path = './network/cvae_graph/' + condition_type + '_condition_train_datasets/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + '/' + str(file_idx) + '.gpickle', 'wb') as f:
            nx.write_gpickle(graph, f)

        with open(save_path + '/' + str(file_idx) + '.pkl', 'wb') as f:
            pickle.dump(building_polygons, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--condition_type", type=str, default="image", help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(condition_type=opt.condition_type)