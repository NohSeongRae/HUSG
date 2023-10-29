import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import numpy as np
import os
from tqdm import tqdm
import pickle


class GraphDataset(Dataset):
    """
    Dataset class for boundary data.
    """
    def __init__(self, data_type='train', n_street=50, n_building=120, n_boundary=200, d_unit=8, d_street=64):
        self.n_street = n_street
        self.n_building = n_building
        self.n_boundary = n_boundary
        self.d_unit = d_unit
        self.d_street = d_street

        load_path = './network/transformer_graph/' + data_type + '_datasets.npz'
        self.full_dataset = np.load(load_path)

        self.unit_position_datasets = self.full_dataset['unit_position_datasets']
        self.street_unit_position_datasets = self.full_dataset['street_unit_position_datasets']
        self.street_index_sequences = self.full_dataset['street_index_sequences']
        self.edge_index_sequences = self.full_dataset['edge_index_sequences']
        self.cur_n_streets = self.full_dataset['cur_n_streets']

        print('unit_position_datasets shape: ', self.unit_position_datasets.shape)
        print('street_unit_position_datasets shape: ', self.street_unit_position_datasets.shape)
        print('street_index_sequences shape: ', self.street_index_sequences.shape)
        print('adj_matrix_sequences shape: ', self.edge_index_sequences.shape)
        print('cur_n_streets shape: ', self.cur_n_streets.shape)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.
        Args:
        - index (int): Index of the item to retrieve.
        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """
        unit_position_dataset = torch.tensor(self.unit_position_datasets[index], dtype=torch.float32)
        street_index_sequence = torch.tensor(self.street_index_sequences[index], dtype=torch.long)
        cur_n_street = torch.tensor(self.cur_n_streets[index], dtype=torch.long)
        edge_index_sequence = torch.tensor(self.edge_index_sequences[index], dtype=torch.long)
        adj_matrix = to_dense_adj(edge_index_sequence)[0].numpy()
        print(adj_matrix)

        # 패딩된 street position 생성
        street_pos = self.street_unit_position_datasets[index]
        street_indices = self.street_index_sequences[index]
        remove_street_indices = np.array([self.n_street + 1, self.n_street + 2, 0])
        street_indices = street_indices[~np.isin(street_indices, remove_street_indices)].astype(int)
        filtered_street_pos = [street_pos[element] for element in street_indices]
        zeros = np.zeros((self.n_boundary, self.d_street, 2))
        zeros[0] = 2
        zeros[1:len(filtered_street_pos) + 1] = filtered_street_pos
        zeros[len(filtered_street_pos) + 1] = 3
        zeros[len(filtered_street_pos) + 2:] = 4
        street_position_dataset = torch.tensor(zeros, dtype=torch.float32)

        # 패딩된 인접 행렬 생성
        zeros = np.zeros((self.n_street + self.n_building, self.n_street + self.n_building))
        zeros[0] = 2
        zeros[1:adj_matrix.shape[0] + 1, 1:adj_matrix.shape[1] + 1] = adj_matrix
        zeros[adj_matrix.shape[0] + 1] = 3
        zeros[adj_matrix.shape[0] + 2:] = 4
        adj_matrix_sequence = torch.tensor(zeros, dtype=torch.float32)
        return unit_position_dataset, street_position_dataset, street_index_sequence, adj_matrix_sequence, cur_n_street


    def __len__(self):
        """
        Get the total number of items in the dataset.
        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)