import torch
import numpy as np
import pickle
from tqdm import tqdm
import os
import networkx as nx

def preprocessing_dataset(data_path='/local_datasets/HUSG/', data_type='train'):
    city_names = ['atlanta']  # Example, use all needed city names for real scenario

    # Paths for the data we'll process to match the DataLoader expectations
    dataset_names = ['adj_matrices', 'node_features', 'building_polygons']

    for city_name in tqdm(city_names):
        adj_matrices = None
        node_features = None
        building_polygons = None

        for dataset_name in dataset_names:
            filepath = './local_datasets/HUSG/atlanta/adj_matrices.pkl'
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Process each dataset accordingly
            if dataset_name == 'adj_matrices':
                adj_matrices = data
            elif dataset_name == 'node_features':
                node_features = data
            elif dataset_name == 'building_polygons':
                building_polygons = data

        for idx in tqdm(range(len(adj_matrices))):
            # Convert edge matrix to edge indices
            edge_indices = np.array(adj_matrices[idx].nonzero()).T

            # Initialize a new combined graph
            combined_graph = nx.Graph()
            combined_graph.add_edges_from(edge_indices)

            # Split the graph into boundary and building based on node features
            boundary_graph = nx.Graph()
            building_graph = nx.Graph()

            for node, features in enumerate(node_features[idx]):
                if features[-1] == 0:  # Assuming the last feature distinguishes boundary from building
                    boundary_graph.add_node(node, boundary_features=features)
                else:
                    building_graph.add_node(node, building_features=features)

            # Adding edges specifically to boundary and building graphs based on the combined graph
            for (u, v) in combined_graph.edges():
                if u in boundary_graph.nodes and v in boundary_graph.nodes:
                    boundary_graph.add_edge(u, v)
                if u in building_graph.nodes and v in building_graph.nodes:
                    building_graph.add_edge(u, v)

            # Save the graphs and features in a structure matching the DataLoader expectations
            combined_graph.graph['boundary_graph'] = boundary_graph
            combined_graph.graph['building_graph'] = building_graph

            # Save the combined graph with boundary and building subgraphs
            save_path = f'./processed_graphs/{data_type}_condition/{city_name}/'
            os.makedirs(save_path, exist_ok=True)
            nx.write_gpickle(combined_graph, os.path.join(save_path, f'graph_{idx}.gpickle'))

            # Save building polygons if necessary
            with open(os.path.join(save_path, f'building_polygons_{idx}.pkl'), 'wb') as f:
                pickle.dump(building_polygons[idx], f)


if __name__ == '__main__':
    preprocessing_dataset(data_path='/local_datasets/HUSG/', data_type='train')