import torch
from torch.utils.data import Dataset
import numpy as np

class BoundaryDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, data_type='train'):
        self.dataset_path = './dataset/husg_transformer_dataset.npz'
        self.dataset = np.load(self.dataset_path)

        self.total_size = len(self.dataset['unit_position_datasets'])
        if data_type == 'train':
            start_index = 0
            end_index = int(self.total_size * train_ratio)
        elif data_type == 'val':
            start_index = int(self.total_size * train_ratio)
            end_index = int(self.total_size * (train_ratio + val_ratio))
        else:
            start_index = int(self.total_size * (train_ratio + val_ratio))
            end_index = int(self.total_size * (train_ratio + val_ratio + test_ratio))

        self.unit_position_datasets = self.dataset['unit_position_datasets'][start_index:end_index]
        self.street_unit_position_datasets = self.dataset['street_unit_position_datasets'][start_index:end_index]
        self.building_index_sequences = self.dataset['building_index_sequences'][start_index:end_index]
        self.street_index_sequences = self.dataset['street_index_sequences'][start_index:end_index]

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
        street_index_sequences = self.street_index_sequences[index]

        return unit_position_dataset, street_position_dataset, building_index_sequence, street_index_sequences

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)