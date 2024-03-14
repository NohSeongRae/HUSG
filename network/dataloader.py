import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle

class GraphDataset(Dataset):
    def __init__(self, data_type='train'):
        """
        Initializes an instance of the GraphDataset class.

        Parameters:
        - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
        """

        super(GraphDataset, self).__init__()

        self.data_type = data_type

        self.folder_path = f'/local_datasets/graph_generation_datasets/{data_type}'
        if data_type == 'test':
            self.folder_path = f'../preprocessing/global_mapper/graph_generation_datasets/{data_type}'

        file_extension = '.pickle'
        count = 0
        try:
            for filename in os.listdir(self.folder_path):
                if filename.endswith(file_extension):
                    count += 1
        except:
            self.folder_path = self.folder_path.replace('/data2', '')
            for filename in os.listdir(self.folder_path):
                if filename.endswith(file_extension):
                    count += 1
        self.pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pickle')]
        self.pkl_files.sort()

        self.data_length = len(self.pkl_files)
        print(self.data_length)

    def pad_matrix(self, matrix, pad_shape):
        """
        Pads a given matrix to the specified shape and returns the padded matrix along with a padding mask.

        Parameters:
        - matrix (np.ndarray): The matrix to be padded.
        - pad_shape (tuple): The desired shape of the padded matrix.

        Returns:
        - padded_matrix (np.ndarray): The matrix padded to the desired shape.
        - pad_mask (np.ndarray): A mask indicating the original data (0s for padded regions and 1s for original data).
        """

        original_shape = matrix.shape
        pad_width = ((0, max(0, pad_shape[0] - original_shape[0])),
                     (0, max(0, pad_shape[1] - original_shape[1])))
        padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)

        pad_mask = np.zeros_like(padded_matrix, dtype=np.float32)
        pad_mask[:original_shape[0], :original_shape[1]] = 1

        return padded_matrix, pad_mask

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the padded matrices and padding masks for boundary adjacency matrix, building adjacency matrix, and boundary-building adjacency matrix, as well as the boundary positions and the number of boundaries and buildings. For test data, it also returns the filename of the loaded pickle file.
        """

        load_path = self.folder_path + '/' + self.pkl_files[idx]
        with open(load_path, 'rb') as f:
            self.data = pickle.load(f)

        boundary_adj_matrix = self.data['boundary_adj_matrix']
        building_adj_matrix = self.data['building_adj_matrix']
        bb_adj_matrix = self.data['bb_adj_matrix']
        boundary_pos_feature = self.data['boundary_pos_feature']
        n_boundary = self.data['n_boundary']
        n_building = self.data['n_building']

        boundary_adj_matrix_padded, boundary_pad_mask = self.pad_matrix(boundary_adj_matrix, (200, 200))
        building_adj_matrix_padded, building_pad_mask = self.pad_matrix(building_adj_matrix, (120, 120))
        bb_adj_matrix_padded, bb_pad_mask = self.pad_matrix(bb_adj_matrix, (120, 200))
        boundary_pos_padded, _ = self.pad_matrix(boundary_pos_feature, (200, 2))

        if self.data_type == 'test':
            return {
                'boundary_adj_matrix_padded': torch.tensor(boundary_adj_matrix_padded, dtype=torch.float32),
                'building_adj_matrix_padded': torch.tensor(building_adj_matrix_padded, dtype=torch.float32),
                'bb_adj_matrix_padded': torch.tensor(bb_adj_matrix_padded, dtype=torch.float32),
                'boundary_pos_padded': torch.tensor(boundary_pos_padded, dtype=torch.float32),
                'boundary_pad_mask': torch.tensor(boundary_pad_mask, dtype=torch.bool)[:, 0],
                'building_pad_mask': torch.tensor(building_pad_mask, dtype=torch.bool)[:, 0],
                'bb_pad_mask': torch.tensor(bb_pad_mask, dtype=torch.bool),
                'n_boundary': n_boundary,
                'n_building': n_building
            }, self.pkl_files[idx]
        else:
            return {
                'boundary_adj_matrix_padded': torch.tensor(boundary_adj_matrix_padded, dtype=torch.float32),
                'building_adj_matrix_padded': torch.tensor(building_adj_matrix_padded, dtype=torch.float32),
                'bb_adj_matrix_padded': torch.tensor(bb_adj_matrix_padded, dtype=torch.float32),
                'boundary_pos_padded': torch.tensor(boundary_pos_padded, dtype=torch.float32),
                'boundary_pad_mask': torch.tensor(boundary_pad_mask, dtype=torch.bool)[:, 0],
                'building_pad_mask': torch.tensor(building_pad_mask, dtype=torch.bool)[:, 0],
                'bb_pad_mask': torch.tensor(bb_pad_mask, dtype=torch.bool),
                'n_boundary': n_boundary,
                'n_building': n_building
            }


    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of items in the dataset.
        """

        return self.data_length