import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import os
import time

class BoundaryDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self, n_boundary, d_street, data_type='train'):
        self.n_boundary = n_boundary
        self.d_street = d_street

        load_path = './network/transformer/' + data_type + '_boundary_datasets.npz'
        self.full_dataset = np.load(load_path)

        self.unit_position_datasets = self.full_dataset['unit_position_datasets']
        self.street_unit_position_datasets = self.full_dataset['street_unit_position_datasets']
        self.street_index_sequences = self.full_dataset['street_index_sequences']
        self.building_exist_sequences = self.full_dataset['building_exist_sequences']

        print(data_type)
        print('unit_position_datasets shape: ', self.unit_position_datasets.shape)
        print('street_unit_position_datasets shape: ', self.street_unit_position_datasets.shape)
        print('street_index_sequences shape: ', self.street_index_sequences.shape)
        print('building_exist_sequences shape: ', self.building_exist_sequences.shape)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """
        unit_position_dataset = self.unit_position_datasets[index]
        street_index_sequence = self.street_index_sequences[index]
        building_exist_sequences = self.building_exist_sequences[index]

        # 패딩된 street position 생성
        street_pos = self.street_unit_position_datasets[index]
        street_indices = self.street_index_sequences[index]
        remove_street_indices = np.array([0])
        street_indices = street_indices[~np.isin(street_indices, remove_street_indices)].astype(int)
        filtered_street_pos = [street_pos[element] for element in street_indices]
        zeros = np.zeros((self.n_boundary, self.d_street, 2))
        zeros[:len(filtered_street_pos)] = filtered_street_pos
        street_position_dataset = torch.tensor(zeros, dtype=torch.float32)

        return unit_position_dataset, street_position_dataset, building_exist_sequences, street_index_sequence

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)