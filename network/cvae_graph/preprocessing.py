import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import networkx as nx
import random

def preprocesing_dataset(condition_type='graph'):

    dataset_path = '../../datasets/sorted_europe_dataset/'

    city_names = ['annecy', 'athens', 'barcelona', 'belgrade',
                  'bologna', 'brasov', 'budapest', 'dublin',
                  'edinburgh', 'firenze', 'lisbon', 'manchester',
                  'milan', 'naples', 'nottingham', 'paris', 'porto',
                  'praha', 'seville', 'stockholm', 'tallinn', 'valencia',
                  'venice', 'verona', 'vienna', 'zurich']

    dataset_names = [
        'edge_indices',
        'node_features',
        'boundary_filenames',
        'building_polygons'
    ]

    file_idx = 0
    for city_name in tqdm(city_names):
        filepath = dataset_path + '/' + city_name + '/' + dataset_names[0] + '.pkl'
        with open(filepath, 'rb') as f:
            edge_indices = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[1] + '.pkl'
        with open(filepath, 'rb') as f:
            node_features = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[2] + '.pkl'
        with open(filepath, 'rb') as f:
            source_file_names = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[3] + '.pkl'
        with open(filepath, 'rb') as f:
            building_polygons = pickle.load(f)

        for idx in range(len(edge_indices)):
            graph = nx.Graph()
            graph.add_edges_from(edge_indices[idx])
            adj_matrix = nx.adjacency_matrix(graph).todense()

            n_node = graph.number_of_nodes()
            n_building = len(building_polygons[idx])
            n_chunk = n_node - n_building

            if np.any((node_features[idx][:, :2] < -0.1) | (node_features[idx][:, :2] > 1.1)):
                print(source_file_names[idx], idx, node_features[idx][:, :2])
                continue

            if condition_type == 'graph':
                street_graph = adj_matrix[:n_chunk, :n_chunk]
                street_graph = nx.DiGraph(street_graph)

                chunk_feature = node_features[idx][:n_chunk]
                for node in street_graph.nodes():
                    street_graph.nodes[node]['chunk_features'] = chunk_feature[node]

                graph.graph['condition'] = street_graph

            for node in graph.nodes():
                graph.nodes[node]['node_features'] = node_features[idx][node]

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
                pickle.dump(building_polygons[idx], f)

            file_idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--condition_type", type=str, default="graph", help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(condition_type=opt.condition_type)