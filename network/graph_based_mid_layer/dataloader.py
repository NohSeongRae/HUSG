import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class VQVAEDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, data_type='train'):
        self.dataset_path = './dataset/husg_vqvae_dataset.npz'

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """
        return np.zeros((3, 128, 128)), np.zeros((3, 128, 128))

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """

        return 128