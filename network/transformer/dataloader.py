import torch
from torch.utils.data import Dataset
import numpy as np

from network.transformer.utils import *

class BoundaryDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self):
        self.src_unit_dataset = []
        self.src_street_dataset = []
        self.trg_dataset = []
        self.gt_one_hot_dataset = []
        self.gt_building_pos_dataset = []
        self.gt_boundary_center_pos_dataset = []

        self.test()

    def test(self):
        """
        Test and populate the dataset with example data.
        """
        street = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
        src_unit_seq = []
        for i in range(len(street) - 1):
            units = interpolate_points(street[i], street[i + 1])
            src_unit_seq.append(units)
        src_unit_seq = np.array(src_unit_seq)
        src_unit_seq = np.reshape(src_unit_seq, (-1, 8, 2))

        src_street_seq = []
        for i in range(len(street) - 1):
            street = sample_points(street[i], street[i + 1], n=32)
            src_street_seq.append(street)
        src_street_seq = np.array(src_street_seq)
        src_street_seq = np.repeat(src_street_seq, repeats=5, axis=0)

        trg_seq = np.ones((src_street_seq.shape[0], 10))
        trg_seq[:, 2:] = 0

        self.src_unit_dataset.append(src_unit_seq)
        self.src_street_dataset.append(src_street_seq)
        self.trg_dataset.append(trg_seq)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """
        src_unit_seq = self.src_unit_dataset[index]
        src_street_seq = self.src_street_dataset[index]
        trg_seq = self.trg_dataset[index]

        return src_unit_seq, src_street_seq, trg_seq

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.src_unit_dataset)
