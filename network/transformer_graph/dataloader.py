import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import os
import pickle


class GraphDataset(torch.utils.data.Dataset):
    """
    Dataset class for boundary data.
    """
    def __init__(self,data_type):
        data_path = './processed_graphs/train_condition/atlanta'
        self.data_type = data_type
        self.load_path = os.path.join(data_path, f'{data_type}_condition', 'atlanta')  # Corrected path
        self.gpickle_files = [f for f in os.listdir(self.load_path) if f.endswith('.gpickle')]
        self.gpickle_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files numerically
        self.data_length = len(self.gpickle_files)  # Total number of files
    def __len__(self):
        return self.data_length
    def get(self, idx):
        #Load the combined graph
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
        node_features=torch.tensor([nx_graph.nodes[node][feature_key] for node in nx_graph.nodes()], dtype=torch.float32)

        # Convert edges
        edge_index = nx.to_scipy_sparse_matrix(nx_graph).tocoo()
        edge_index=torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

        # Create PyG Data object
        data=Data(x=node_features, edge_index=edge_index, num_nodes=nx_graph.number_of_nodes())
        return data

    def create_boundary_building_adjmatrix(self, boundary_graph, building_graph,combined_edge_index):
        # Get node indices for boundary and building graphs
        boundary_indices = {node: i for i, node in enumerate(boundary_graph.nodes())}
        building_indices = {node: i + len(boundary_indices) for i, node in enumerate(building_graph.nodes())}

        # Initialize adjacency matrix
        num_boundary = len(boundary_indices)
        num_building = len(building_indices)
        adj_matrix = torch.zeros((num_boundary, num_building), dtype=torch.float32)

        # Fill adjacenecy matrix for boundary-building relationships
        # This part depends on how the relationships are defined in your data
        # For example, if they're stored as edge attributes in the combined graph:

        # Process the combined edge index
        # Assuming combined_edge_index is a 2xN tensor where rows are edges (u, v)
        for edge in combined_edge_index.t():
            u, v = edge[0].item(), edge[1].item()  # Convert tensor to integer
            # Check if one node is in the boundary and the other in the building
            if u in boundary_indices and v in building_indices:
                adj_matrix[boundary_indices[u], building_indices[v]] = 1
            elif v in boundary_indices and u in building_indices:
                adj_matrix[boundary_indices[v], building_indices[u]] = 1

        return adj_matrix