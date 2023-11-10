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
    # city_names = ["atlanta"]

    dataset_names = [
        'edge_indices',
        'node_features',
        'building_semantics',
        'insidemask'
    ]

    graphs = []

    for city_name in tqdm(city_names):
        filepath = dataset_path + '/' + city_name + '/' + dataset_names[0] + '.pkl'
        with open(filepath, 'rb') as f:
            edge_indices = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[1] + '.pkl'
        with open(filepath, 'rb') as f:
            node_features = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[2] + '.pkl'
        with open(filepath, 'rb') as f:
            building_semantics = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[3] + '.pkl'
        with open(filepath, 'rb') as f:
            inside_masks = pickle.load(f)

        for idx in range(len(edge_indices)):
            graph = nx.Graph()
            graph.add_edges_from(edge_indices[idx])
            adj_matrix = nx.adjacency_matrix(graph).todense()

            n_node = graph.number_of_nodes()
            n_building = len(building_semantics[idx])
            n_chunk = n_node - n_building

            if np.any((node_features[idx][:, :2] < -0.1) | (node_features[idx][:, :2] > 1.1)):
                print(idx, node_features[idx][:, :2])
                continue

            if condition_type == 'image':
                graph.graph['condition'] = inside_masks[idx]

            elif condition_type == 'graph':
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

            semantic_list = [["shop", "supermarket", "restaurant", "tourism", "accommodation"],
                             ["kindergarten", "school", "college", "university"],
                             ["police_station", "ambulance_station", "fire_station"],
                             ["bank", "bureau_de_change"],
                             ["government_office", "embassy", "military", "post_office"],
                             ["doctor", "dentist", "clinic", "hospital", "pharmacy", "alternative"],
                             ["place_of_worship", "community_centre", "library", "historic", "toilet"],
                             ["stadium", "swimming_pool", "pitch", "sport_centre"],
                             ['residence']]

            for node in graph.nodes():
                if node < n_chunk:
                    graph.nodes[node]['node_semantics'] = 0
                else:
                    for i in range(len(semantic_list)):
                        if building_semantics[idx][node - n_chunk] in semantic_list[i]:
                            graph.nodes[node]['node_semantics'] = i + 1
                            break
                print(city_name, idx)
                print(building_semantics[idx])
                print(graph.nodes[node]['node_semantics'], n_node, n_chunk, n_building)
            graphs.append(graph)

    random.shuffle(graphs)

    total_size = len(graphs)

    for data_type in tqdm(['train', 'val', 'test']):
        if data_type == 'train':
            start_index = 0
            end_index = int(total_size * train_ratio)
        elif data_type == 'val':
            start_index = int(total_size * train_ratio)
            end_index = int(total_size * (train_ratio + val_ratio))
        else:
            start_index = int(total_size * (train_ratio + val_ratio))
            end_index = int(total_size * (train_ratio + val_ratio + test_ratio))

        save_path = './network/cvae_graph/' + condition_type + '_condition_train_datasets/'+ data_type
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for idx in range(start_index, end_index):
            with open(save_path + '/' + str(idx - start_index) + '.gpickle', 'wb') as f:
                nx.write_gpickle(graphs[idx], f)

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
    parser.add_argument("--condition_type", type=str, default="graph", help="Random seed for reproducibility across runs.")

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