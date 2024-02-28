import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import numpy as np
import os
import pickle


class GraphDataset(Dataset):
    """
    Dataset class for boundary data.
    """
    def __init__(self, data_type='train', n_building=120, n_boundary=200):
        # self.n_street = n_street
        self.n_building = n_building
        self.n_boundary = n_boundary

        # self.d_unit = d_unit
        # self.d_street = d_street

        load_path = './network/transformer_graph/' + data_type + '_datasets.npz'

        #debug
        try:
            self.full_dataset = np.load(load_path, allow_pickle=True)

            # 필수 필드 확인
            required_fields = ['boundary_adj_matrices', 'building_adj_matrices', 'inter_graph_adj_matrices']
            for field in required_fields:
                if field in self.full_dataset:
                    setattr(self, field, self.full_dataset[field])
                    print(f"{field} loaded successfully.")
                else:
                    print(f"Error: '{field}' is missing in the dataset.")
        except FileNotFoundError:
            print(f"Error: The file {load_path} does not exist.")



        # Individual graphs
        self.boundary_adj_matrices = self.full_dataset['boundary_adj_matrices']
        self.building_adj_matrices = self.full_dataset['building_adj_matrices']

        # Ground truth adjacency matrix for connections between boundary and building graphs
        self.inter_graph_adj_matrices = self.full_dataset['inter_graph_adj_matrices']

        if data_type is not 'test':
            if torch.distributed.get_rank() == 1:
                print('boundary_adj_matrices shape: ', self.boundary_adj_matrices.shape)
                print('building_adj_matrices shape: ', self.building_adj_matrices.shape)
                print('inter_graph_adj_matrices shape: ', self.inter_graph_adj_matrices.shape)


    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.
        Args:
        - index (int): Index of the item to retrieve.
        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """

        boundary_adj_matrix = torch.tensor(self.boundary_adj_matrices[index], dtype=torch.float32)
        building_adj_matrix = torch.tensor(self.building_adj_matrices[index], dtype=torch.float32)

        torch.fill_diagonal_(boundary_adj_matrix, 0)
        torch.fill_diagonal_(building_adj_matrix, 0)

        inter_graph_adj_matrix = torch.tensor(self.inter_graph_adj_matrices[index], dtype=torch.float32)

        return boundary_adj_matrix, building_adj_matrix, inter_graph_adj_matrix




    def __len__(self):
        """
        Get the total number of items in the dataset.
        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.building_adj_matrices)