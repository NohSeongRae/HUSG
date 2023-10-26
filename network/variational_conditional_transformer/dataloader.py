import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import pickle

class BoundaryDataset(Dataset):
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
        ]

        n_street = 50
        n_building = 120
        n_boundary = 250
        d_unit = 8
        d_street = 64

        all_unit_position_datasets = []
        all_street_unit_position_datasets = []
        all_street_index_sequences = []
        all_gt_unit_position_datasets = []

        for city_name in tqdm(city_names):
            loaded_data = {}

            for dataset_name in dataset_names:
                filepath = dataset_path + '/' + city_name + '/' + dataset_name + '.pkl'
                with open(filepath, 'rb') as f:
                    loaded_data[dataset_name] = pickle.load(f)
                    for idx in range(len(loaded_data[dataset_name])):
                        data = loaded_data[dataset_name][idx]
                        if dataset_name == 'street_index_sequences':
                            zeros = np.zeros(n_boundary)
                            zeros[:len(data)] = data
                            data = zeros
                            all_street_index_sequences.append(data)

                        elif dataset_name == 'unit_position_datasets':
                            zeros = np.zeros((n_boundary, 2, 2))
                            p1 = np.reshape(np.array(data[:, 0, :]), (-1, 1, 2))
                            p2 = np.reshape(np.array(data[:, -1, :]), (-1, 1, 2))
                            gt = np.concatenate((p1, p2), axis=1)
                            zeros[:len(gt)] = gt
                            zeros = np.reshape(zeros, (-1, 4))
                            all_gt_unit_position_datasets.append(zeros)

                            zeros = np.zeros((n_boundary, d_unit, 2))
                            zeros[:len(data)] = data
                            data = zeros
                            all_unit_position_datasets.append(data)

                        elif dataset_name == 'street_unit_position_datasets':
                            zeros = np.zeros((n_boundary, d_street, 2))
                            zeros[:len(data)] = data
                            data = zeros
                            all_street_unit_position_datasets.append(data)

        # Concatenate data from all cities for each key
        cls.full_dataset = {
            'unit_position_datasets': np.array(all_unit_position_datasets),
            'street_unit_position_datasets': np.array(all_street_unit_position_datasets),
            'street_index_sequences': np.array(all_street_index_sequences),
            'gt_unit_position_datasets': np.array(all_gt_unit_position_datasets)
        }

        # Shuffle the dataset
        permuted_indices = np.random.permutation(len(cls.full_dataset['unit_position_datasets']))
        for key in cls.full_dataset:
            cls.full_dataset[key] = cls.full_dataset[key][permuted_indices]

    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, data_type='train', load=True):
        if self.full_dataset is None:
            if load:
                self.load_full_dataset()
                save_path = './network/variational_conditional_transformer/datasets'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.savez(save_path,
                         unit_position_datasets=self.full_dataset['unit_position_datasets'],
                         street_unit_position_datasets=self.full_dataset['street_unit_position_datasets'],
                         street_index_sequences=self.full_dataset['street_index_sequences'],
                         gt_unit_position_datasets=self.full_dataset['gt_unit_position_datasets'])
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
        self.gt_unit_position_datasets = self.full_dataset['gt_unit_position_datasets'][self.start_index:self.end_index]

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