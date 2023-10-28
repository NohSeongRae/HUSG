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

    def __init__(self, data_type='train'):
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
        street_position_dataset = self.street_unit_position_datasets[index]
        street_index_sequence = self.street_index_sequences[index]
        gt_unit_position_dataset = self.gt_unit_position_datasets[index]

        return unit_position_dataset, street_position_dataset, street_index_sequence, gt_unit_position_dataset

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)