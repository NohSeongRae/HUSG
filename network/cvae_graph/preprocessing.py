import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import networkx as nx
import random

def preprocesing_dataset(condition_type='graph'):

    dataset_path = '../../datasets/HUSG/'

    city_names = ['annecy', 'athens', 'barcelona', 'belgrade',
                  'bologna', 'brasov', 'budapest', 'dublin',
                  'edinburgh', 'firenze', 'lisbon', 'manchester',
                  'milan', 'naples', 'nottingham', 'paris', 'porto',
                  'praha', 'seville', 'stockholm', 'tallinn', 'valencia',
                  'venice', 'verona', 'vienna', 'zurich']

    dataset_names = [
        'edge_indices',
        'node_features',
        'building_semantics',
        'insidemask',
        'boundary_filenames',
        'insidemask_filename',
        'building_polygons',
        'boundarymask'
    ]

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

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[4] + '.pkl'
        with open(filepath, 'rb') as f:
            source_file_names = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[5] + '.pkl'
        with open(filepath, 'rb') as f:
            mask_file_names = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[6] + '.pkl'
        with open(filepath, 'rb') as f:
            building_polygons = pickle.load(f)

        filepath = dataset_path + '/' + city_name + '/' + dataset_names[7] + '.pkl'
        with open(filepath, 'rb') as f:
            boundary_masks = pickle.load(f)

        idx = 0
        while idx < len(edge_indices):
            if source_file_names[idx].split('/')[-1] != mask_file_names[idx]:
                del mask_file_names[idx]
                del inside_masks[idx]
                del boundary_masks[idx]

            else:
                idx += 1

        for idx in range(len(edge_indices)):
            graph = nx.Graph()
            graph.add_edges_from(edge_indices[idx])
            adj_matrix = nx.adjacency_matrix(graph).todense()

            n_node = graph.number_of_nodes()
            n_building = len(building_semantics[idx])
            n_chunk = n_node - n_building

            if np.any((node_features[idx][:, :2] < -0.1) | (node_features[idx][:, :2] > 1.1)):
                print(source_file_names[idx], idx, node_features[idx][:, :2])
                continue

            if condition_type == 'image':
                inside_masks[idx] = np.flipud(inside_masks[idx])
                graph.graph['condition'] = inside_masks[idx]

            elif condition_type == 'image_resnet34':
                inside_masks[idx] = np.flipud(inside_masks[idx])
                boundary_masks[idx] = np.flipud(boundary_masks[idx])
                zero_channel = np.zeros((224, 224))
                combined_mask = np.stack([inside_masks[idx], boundary_masks[idx], zero_channel], axis=0)
                graph.graph['condition'] = combined_mask

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

            if 'cemetery' in building_semantics[idx][:] or 'seating' in building_semantics[idx][:]:
                continue

            for node in graph.nodes():
                if node < n_chunk:
                    graph.nodes[node]['node_semantics'] = 0
                else:
                    for i in range(len(semantic_list)):
                        if building_semantics[idx][node - n_chunk] in semantic_list[i]:
                            graph.nodes[node]['node_semantics'] = i + 1
                            break

            save_path = './network/cvae_graph/' + condition_type + '_condition_train_datasets/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with open(save_path + '/' + mask_file_names[idx] + '.gpickle', 'wb') as f:
                nx.write_gpickle(graph, f)

            with open(save_path + '/' + mask_file_names[idx] + '.pkl', 'wb') as f:
                pickle.dump(building_polygons[idx], f)


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