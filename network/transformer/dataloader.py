import torch
from torch.utils.data import Dataset
import numpy as np

from network.transformer.utils import *

class BoundaryDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self):
        self.dataset_path = './dataset/husg_transformer_dataset.npz'
        self.dataset = np.load(self.dataset_path)
        self.unit_position_datasets = self.dataset['unit_position_datasets']
        self.street_unit_position_datasets = self.dataset['street_unit_position_datasets']
        self.one_hot_building_index_sequences = self.dataset['one_hot_building_index_sequences']
        self.building_index_sequences = self.dataset['building_index_sequences']
        self.building_index_sequences = self.dataset['building_index_sequences']
        self.street_index_sequences = self.dataset['street_index_sequences']
        self.building_center_position_datasets = self.dataset['building_center_position_datasets']
        self.unit_center_position_datasets = self.dataset['unit_center_position_datasets']

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
        building_index_sequence = self.building_index_sequences[index]
        one_hot_building_index_sequence = self.one_hot_building_index_sequences[index]
        building_center_position_dataset = self.building_center_position_datasets[index]
        unit_center_position_dataset = self.unit_center_position_datasets[index]


        return unit_position_dataset, street_position_dataset, building_index_sequence, \
               one_hot_building_index_sequence, building_center_position_dataset, unit_center_position_dataset

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)
