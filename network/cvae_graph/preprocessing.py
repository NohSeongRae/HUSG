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

    # city_names = ['annecy', 'athens', 'barcelona', 'belgrade',
    #               'bologna', 'brasov', 'budapest', 'dublin',
    #               'edinburgh', 'firenze', 'lisbon', 'manchester',
    #               'milan', 'naples', 'nottingham', 'paris', 'porto',
    #               'praha', 'seville', 'stockholm', 'tallinn', 'valencia',
    #               'venice', 'verona', 'vienna', 'zurich']
    city_names = ['atlanta']

    dataset_names = [
        'edge_indices',
        'node_features',
        'building_polygons',
    ]

    for city_name in tqdm(city_names):
        # Load necessary files
        edge_indices = pickle.load(open(os.path.join(dataset_path, city_name, 'edge_indices.pkl'), 'rb'))
        node_features = pickle.load(open(os.path.join(dataset_path, city_name, 'node_features.pkl'), 'rb'))
        building_polygons = pickle.load(open(os.path.join(dataset_path, city_name, 'building_polygons.pkl'), 'rb'))
        # Process each graph in the city dataset
        for idx, edges in enumerate(edge_indices):
            graph = nx.Graph()
            graph.add_edges_from(edges)

            # Assign node features
            for node in range(len(node_features[idx])):
                graph.nodes[node]['node_features'] = node_features[idx][node]

            # Create and assign building masks
            zeros = np.zeros((graph.number_of_nodes(), 1))
            n_building = len(building_polygons[idx])  # Number of buildings
            zeros[-n_building:] = 1  # Assuming the last nodes are buildings
            for node in graph.nodes():
                graph.nodes[node]['building_masks'] = zeros[node, 0]

            # Save processed graph
            save_path = f'./network/cvae_graph/{condition_type}_condition_train_datasets/{city_name}'
            os.makedirs(save_path, exist_ok=True)
            nx.write_gpickle(graph, os.path.join(save_path, f'{idx}.gpickle'))
            pickle.dump(building_polygons[idx], open(os.path.join(save_path, f'{idx}_buildings.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset for Graph Convolutional Networks.")
    parser.add_argument("--condition_type", type=str, default="graph", help="Type of condition for the dataset.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    preprocesing_dataset(condition_type=args.condition_type)