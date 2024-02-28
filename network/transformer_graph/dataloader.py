import torch
import networkx as nx
from torch_geometric.data import Data, Dataset
import numpy as np
import os
import pickle


class GraphDataset(torch.utils.data.Dataset):
    """
    Dataset class for boundary data.
    """
    def __init__(self,data_type):
        data_path = '/data/ssw03270/datasets/'
        self.data_type = data_type
        self.load_path = os.path.join(data_path,  'atlanta')  # Corrected path
        if not os.path.exists(self.load_path):
            raise FileNotFoundError(f"The specified directory does not exist: {self.load_path}")

        self.gpickle_files = [f for f in os.listdir(self.load_path) if f.endswith('.gpickle')]
        if not self.gpickle_files:
            raise FileNotFoundError(f"No .gpickle files found in directory: {self.load_path}")
        self.gpickle_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files numerically
        self.data_length = len(self.gpickle_files)  # Total number of files
    def __len__(self):
        return self.data_length
    def __getitem__(self, idx):
        #Load the combined graph
        if idx >= self.data_length or idx < 0:  # Check index bounds
            raise IndexError(f"Index {idx} out of range for dataset of size {self.data_length}.")
        load_path=os.path.join(self.load_path, self.gpickle_files[idx]) #path including data filename
        with open(load_path, 'rb') as f:
            combined_graph=pickle.load(f)

        # Extract boundary and building graphs
        boundary_graph = combined_graph.graph['boundary_graph']
        building_graph = combined_graph.graph['building_graph']

        # Extract the combined edge index
        # Ensure this line matches how you store edges in your combined_graph data structure
        combined_edge_index = torch.tensor(np.array(list(combined_graph.edges())).T, dtype=torch.long)

        # Convert NetworkX graphs to PyTorch_Geometric Data Objects
        boundary_data = self.convert_to_pyg_data(boundary_graph, feature_key='boundary_features')
        building_data = self.convert_to_pyg_data(building_graph, feature_key='building_features')

        # Create adjacency matrix for the boundary-building relationships
        # Pass the combined edge index to the function
        adj_matrix = self.create_boundary_building_adjmatrix(boundary_graph, building_graph, combined_edge_index)

        return boundary_data, building_data, adj_matrix
    def convert_to_pyg_data(self, nx_graph, feature_key):
        #Convert node features
        node_features_list = [nx_graph.nodes[node][feature_key] for node in nx_graph.nodes()]
        node_features_array = np.array(node_features_list)  # Convert list of numpy arrays to single numpy array
        node_features = torch.tensor(node_features_array, dtype=torch.float32)  # Convert numpy array to tensor

        # Convert edges
        edge_index = nx.to_scipy_sparse_matrix(nx_graph).tocoo()
        edge_index=torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

        # Create PyG Data object
        data=Data(x=node_features, edge_index=edge_index, num_nodes=nx_graph.number_of_nodes())
        return data

    def create_boundary_building_adjmatrix(self, boundary_graph, building_graph, combined_edge_index):
        # Maps for node indices
        boundary_indices = {node: i for i, node in enumerate(boundary_graph.nodes())}
        building_indices = {node: i for i, node in enumerate(building_graph.nodes())}

        # Initialize adjacency matrix
        num_boundary = len(boundary_indices)
        num_building = len(building_indices)
        adj_matrix = torch.zeros((num_boundary, num_building), dtype=torch.float32)

        # Ensure that each edge is valid before attempting to add it to the adjacency matrix
        for edge in combined_edge_index.t():
            u, v = edge[0].item(), edge[1].item()  # Extract node indices

            if u in boundary_indices and v in building_indices:
                # Check within bounds before assigning
                if boundary_indices[u] < num_boundary and building_indices[v] < num_building:
                    adj_matrix[boundary_indices[u], building_indices[v]] = 1
                else:
                    # Log or handle the error appropriately here
                    print(f"Error: Index out of bounds for boundary-building matrix at edge ({u}, {v}).")
            elif v in boundary_indices and u in building_indices:
                # Check within bounds before assigning
                if boundary_indices[v] < num_boundary and building_indices[u] < num_building:
                    adj_matrix[boundary_indices[v], building_indices[u]] = 1
                else:
                    # Log or handle the error appropriately here
                    print(f"Error: Index out of bounds for boundary-building matrix at edge ({v}, {u}).")

        return adj_matrix