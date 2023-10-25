import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm

class GraphDataset(Dataset):
    """
    Dataset class for boundary data.
    """
    # Load and store the entire dataset in class variables
    full_dataset = None

    @classmethod
    def load_full_dataset(cls):
        dataset_path = '../../datasets/HUSG/'
        city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
                      "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
                      "sanfrancisco", "miami", "seattle", "boston", "providence",
                      "neworleans", "denver", "pittsburgh", "tampa", "washington"]
        city_names = ['atlanta']
        file_name = '/husg_transformer_graph_dataset.npz'

        if cls.full_dataset is None:
            all_unit_position_datasets = []
            all_street_unit_position_datasets = []
            all_street_index_sequences = []
            all_adj_matrix_sequences = []

            for city_name in tqdm(city_names):
                data = np.load(dataset_path + city_name + file_name)

                all_unit_position_datasets.append(data['unit_position_datasets'])
                all_street_unit_position_datasets.append(data['street_unit_position_datasets'])
                all_street_index_sequences.append(data['street_index_sequences'])
                all_adj_matrix_sequences.append(data['adj_matrices'])

            # Concatenate data from all cities for each key
            cls.full_dataset = {
                'unit_position_datasets': np.concatenate(all_unit_position_datasets, axis=0),
                'street_unit_position_datasets': np.concatenate(all_street_unit_position_datasets, axis=0),
                'street_index_sequences': np.concatenate(all_street_index_sequences, axis=0),
                'adj_matrix_sequences': np.concatenate(all_adj_matrix_sequences, axis=0)
            }

    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, data_type='train', load=True):
        if self.full_dataset is None:
            if load:
                self.load_full_dataset()
                save_path = './network/transformer_graph/datasets'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.savez(save_path,
                         unit_position_datasets=self.full_dataset['unit_position_datasets'],
                         street_unit_position_datasets=self.full_dataset['street_unit_position_datasets'],
                         street_index_sequences=self.full_dataset['street_index_sequences'],
                         adj_matrix_sequences=self.full_dataset['adj_matrix_sequences'])
        else:
            load_path = './network/transformer_graph/datasets.npz'
            self.full_dataset = np.load(load_path)

        total_size = len(self.full_dataset['unit_position_datasets'])
        if data_type == 'train':
            self.start_index = 0
            self.end_index = int(total_size * train_ratio)
        elif data_type == 'val':
            self.start_index = int(total_size * train_ratio)
            self.end_index = int(total_size * (train_ratio + val_ratio))
        else:
            self.start_index = int(total_size * (train_ratio + val_ratio))
            self.end_index = int(total_size * (train_ratio + val_ratio + test_ratio))

        self.unit_position_datasets = self.full_dataset['unit_position_datasets'][self.start_index:self.end_index]
        self.street_unit_position_datasets = self.full_dataset['street_unit_position_datasets'][self.start_index:self.end_index]
        self.street_index_sequences = self.full_dataset['street_index_sequences'][self.start_index:self.end_index]
        self.adj_matrix_sequences = self.full_dataset['adj_matrix_sequences'][self.start_index:self.end_index]

        print(data_type)
        print('unit_position_datasets shape: ', self.unit_position_datasets.shape)
        print('street_unit_position_datasets shape: ', self.street_unit_position_datasets.shape)
        print('street_index_sequences shape: ', self.street_index_sequences.shape)
        print('adj_matrix_sequences shape: ', self.adj_matrix_sequences.shape)

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
        adj_matrix_sequence = self.adj_matrix_sequences[index]

        return unit_position_dataset, street_position_dataset, street_index_sequence, adj_matrix_sequence

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)