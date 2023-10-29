import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import pickle

class BoundaryDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self, sos_idx, eos_idx, pad_idx, n_boundary, n_street, d_street, data_type='train'):
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.n_boundary = n_boundary
        self.n_street = n_street
        self.d_street = d_street

        load_path = './network/variational_conditional_transformer/' + data_type + '_boundary_datasets.npz'
        self.full_dataset = np.load(load_path)

        self.unit_position_datasets = self.full_dataset['unit_position_datasets']
        self.street_unit_position_datasets = self.full_dataset['street_unit_position_datasets']
        self.street_index_sequences = self.full_dataset['street_index_sequences']
        self.gt_unit_position_datasets = self.full_dataset['gt_unit_position_datasets']

        print('unit_position_datasets shape: ', self.unit_position_datasets.shape)
        print('street_unit_position_datasets shape: ', self.street_unit_position_datasets.shape)
        print('street_index_sequences shape: ', self.street_index_sequences.shape)
        print('gt_unit_position_datasets shape: ', self.gt_unit_position_datasets.shape)

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
        gt_unit_position_dataset = self.gt_unit_position_datasets[index]

        # 패딩된 street position 생성
        street_pos = self.street_unit_position_datasets[index]
        street_indices = self.street_index_sequences[index]
        remove_street_indices = np.array([0, self.n_street + 1, self.n_street + 2])
        street_indices = street_indices[~np.isin(street_indices, remove_street_indices)].astype(int)
        filtered_street_pos = [street_pos[element] for element in street_indices]
        zeros = np.zeros((self.n_boundary, self.d_street, 2))
        zeros[0] = self.sos_idx
        zeros[1:len(filtered_street_pos) + 1] = filtered_street_pos
        zeros[len(filtered_street_pos) + 1] = self.eos_idx
        zeros[len(filtered_street_pos) + 2] = self.pad_idx
        street_position_dataset = torch.tensor(zeros, dtype=torch.float32)

        return unit_position_dataset, street_position_dataset, street_index_sequence, gt_unit_position_dataset

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)