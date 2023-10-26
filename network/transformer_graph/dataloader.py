import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import pickle

class GraphDataset(Dataset):
    """
    Dataset class for boundary data.
    """
    # Load and store the entire dataset in class variables
    full_dataset = None

    @classmethod
    def load_full_dataset(cls):
        dataset_path = '../../datasets/HUSG/'
        # dataset_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'train_dataset')

        # city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
        #               "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
        #               "sanfrancisco", "miami", "seattle", "boston", "providence",
        #               "neworleans", "denver", "pittsburgh", "tampa", "washington"]
        city_names = ['atlanta', 'dallas']

        dataset_names = [
            'street_index_sequences',
            'unit_position_datasets',
            'street_unit_position_datasets',
            'adj_matrices'
        ]

        n_street = 50
        n_building = 120
        n_boundary = 250
        d_unit = 8
        d_street = 64

        all_unit_position_datasets = []
        all_street_unit_position_datasets = []
        all_street_index_sequences = []
        all_adj_matrix_sequences = []
        all_cur_n_street = []

        for city_name in tqdm(city_names):
            loaded_data = {}

            for dataset_name in dataset_names:
                filepath = dataset_path + '/' + city_name + '/' + dataset_name + '.pkl'
                with open(filepath, 'rb') as f:
                    loaded_data[dataset_name] = pickle.load(f)
                    for idx in range(len(loaded_data[dataset_name])):
                        data = loaded_data[dataset_name][idx]
                        if dataset_name == 'street_index_sequences':
                            cur_n_street = np.max(data, axis=0)
                            all_cur_n_street.append(cur_n_street)

                            zeros = np.zeros(n_boundary)
                            zeros[0] = n_street + 1
                            zeros[1:len(data) + 1] = data
                            zeros[len(data) + 1] = n_street + 2
                            data = zeros
                            all_street_index_sequences.append(data)

                        elif dataset_name == 'unit_position_datasets':
                            zeros = np.zeros((n_boundary, d_unit, 2))
                            zeros[0] = 2
                            zeros[1:len(data) + 1] = data
                            zeros[len(data) + 1] = 3
                            zeros[len(data) + 2:] = 4
                            data = zeros
                            all_unit_position_datasets.append(data)

                        elif dataset_name == 'street_unit_position_datasets':
                            zeros = np.zeros((n_boundary, d_street, 2))
                            zeros[0] = 2
                            zeros[1:len(data) + 1] = data
                            zeros[len(data) + 1] = 3
                            zeros[len(data) + 2:] = 4
                            data = zeros
                            all_street_unit_position_datasets.append(data)

                        elif dataset_name == 'adj_matrices':
                            zeros = np.zeros((n_street + n_building, n_street + n_building))
                            zeros[0] = 2
                            zeros[1:len(data) + 1, 1:len(data)+1] = data
                            zeros[len(data) + 1] = 3
                            zeros[len(data) + 2:] = 4
                            data = zeros
                            all_adj_matrix_sequences.append(data)

        # Concatenate data from all cities for each key
        cls.full_dataset = {
            'unit_position_datasets': np.array(all_unit_position_datasets),
            'street_unit_position_datasets': np.array(all_street_unit_position_datasets),
            'street_index_sequences': np.array(all_street_index_sequences),
            'adj_matrix_sequences': np.array(all_adj_matrix_sequences),
            'cur_n_streets': np.array(all_cur_n_street)
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
                         adj_matrix_sequences=self.full_dataset['adj_matrix_sequences'],
                         cur_n_streets=self.full_dataset['cur_n_streets'])
        else:
            load_path = './network/transformer_graph/datasets.npz'
            loaded_data = np.load(load_path)
            self.full_dataset = {key: loaded_data[key] for key in loaded_data.files}

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
        self.cur_n_streets = self.full_dataset['cur_n_streets'][self.start_index:self.end_index]

        print('unit_position_datasets shape: ', self.unit_position_datasets.shape)
        print('street_unit_position_datasets shape: ', self.street_unit_position_datasets.shape)
        print('street_index_sequences shape: ', self.street_index_sequences.shape)
        print('adj_matrix_sequences shape: ', self.adj_matrix_sequences.shape)
        print('cur_n_streets shape: ', self.cur_n_streets.shape)

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
        cur_n_street = self.cur_n_streets[index]

        return unit_position_dataset, street_position_dataset, street_index_sequence, adj_matrix_sequence, cur_n_street

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)