import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class MidLayerDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, data_type='train'):
        self.dataset_path = './dataset/husg_mid_layer_dataset.npz'
        # self.dataset = np.load(self.dataset_path)
        #
        # self.total_size = len(self.dataset['boundary_masks'])
        # if data_type == 'train':
        #     self.start_index = 0
        #     self.end_index = int(self.total_size * train_ratio)
        # elif data_type == 'val':
        #     self.start_index = int(self.total_size * train_ratio)
        #     self.end_index = int(self.total_size * (train_ratio + val_ratio))
        # else:
        #     self.start_index = int(self.total_size * (train_ratio + val_ratio))
        #     self.end_index = int(self.total_size * (train_ratio + val_ratio + test_ratio))
        #
        # self.boundary_masks = self.dataset['boundary_masks'][self.start_index:self.end_index]
        # self.inside_masks = self.dataset['inside_masks'][self.start_index:self.end_index]
        # self.street_masks = self.dataset['street_masks'][self.start_index:self.end_index]
        # self.all_masks = self.dataset['all_masks'][self.start_index:self.end_index]
        # self.tf_output_seq_masks = self.dataset['tf_output_seq_masks'][self.start_index:self.end_index]
        # self.tf_output_plot_masks = self.dataset['tf_output_plot_masks'][self.start_index:self.end_index]
        # self.in_building_cp_masks = self.dataset['in_building_cp_masks'][self.start_index:self.end_index]
        # self.in_edge_masks = self.dataset['in_edge_masks'][self.start_index:self.end_index]

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """
        return np.zeros((8, 128, 128)), np.zeros((2, 128, 128))

        boundary_mask = self.boundary_masks[index]
        inside_mask = self.inside_masks[index]
        street_mask = self.street_masks[index]
        all_mask = self.all_masks[index]
        tf_output_seq_mask = self.tf_output_seq_masks[index]
        tf_output_plot_mask = self.tf_output_plot_masks[index]
        in_building_cp_mask = self.in_building_cp_masks[index]
        in_edge_mask = self.in_edge_masks[index]

        data = np.concatenate((boundary_mask, inside_mask, street_mask, all_mask,
                               tf_output_seq_mask, tf_output_plot_mask,
                               in_building_cp_mask, in_edge_mask), axis=0)

        return data

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """

        return 128
        return len(self.boundary_masks)