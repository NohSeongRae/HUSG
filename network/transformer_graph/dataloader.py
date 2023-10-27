import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse, to_dense_adj
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

        city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
                      "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
                      "sanfrancisco", "miami", "seattle", "boston", "providence",
                      "neworleans", "denver", "pittsburgh", "tampa", "washington"]
        # city_names = ["atlanta", "boston", "dallas", "denver", "houston", "lasvegas",
        #               "littlerock", "miami", "neworleans"]

        dataset_names = [
            'street_index_sequences',
            'unit_position_datasets',
            'street_unit_position_datasets',
            'adj_matrices'
        ]

        n_street = 70
        n_building = 120
        n_boundary = 250
        d_unit = 8
        d_street = 64

        all_unit_position_datasets = []
        all_street_unit_position_datasets = []
        all_street_index_sequences = []
        all_edge_index_sequences = []
        all_cur_n_street = []
        max_edge_length = 0

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
                            zeros = np.zeros((n_street, d_street, 2))
                            data = np.unique(data, axis=0)
                            zeros[1:len(data)+1] = data
                            data = zeros
                            all_street_unit_position_datasets.append(data)

                        elif dataset_name == 'adj_matrices':
                            edge_index, _ = dense_to_sparse(torch.tensor(data, dtype=torch.float32))
                            max_edge_length = max(max_edge_length, edge_index.size(1))
                            all_edge_index_sequences.append(edge_index.numpy())

        for i in range(len(all_edge_index_sequences)):
            padded_edge_index = np.pad(all_edge_index_sequences[i],
                                       ((0, 0), (0, max_edge_length - all_edge_index_sequences[i].shape[1])),
                                       mode='constant', constant_values=0)
            all_edge_index_sequences[i] = padded_edge_index

        # Concatenate data from all cities for each key
        cls.full_dataset = {
            'unit_position_datasets': np.array(all_unit_position_datasets),
            'street_unit_position_datasets': np.array(all_street_unit_position_datasets),
            'street_index_sequences': np.array(all_street_index_sequences),
            'edge_index_sequences': np.array(all_edge_index_sequences),
            'cur_n_streets': np.array(all_cur_n_street)
        }

        # Shuffle the dataset
        permuted_indices = np.random.permutation(len(cls.full_dataset['unit_position_datasets']))
        for key in cls.full_dataset:
            cls.full_dataset[key] = cls.full_dataset[key][permuted_indices]

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
                         edge_index_sequences=self.full_dataset['edge_index_sequences'],  # 수정된 부분
                         cur_n_streets=self.full_dataset['cur_n_streets'])
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
        self.edge_index_sequences = self.full_dataset['edge_index_sequences'][self.start_index:self.end_index]
        self.cur_n_streets = self.full_dataset['cur_n_streets'][self.start_index:self.end_index]

        print('unit_position_datasets shape: ', self.unit_position_datasets.shape)
        print('street_unit_position_datasets shape: ', self.street_unit_position_datasets.shape)
        print('street_index_sequences shape: ', self.street_index_sequences.shape)
        print('adj_matrix_sequences shape: ', self.edge_index_sequences.shape)
        print('cur_n_streets shape: ', self.cur_n_streets.shape)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at a specific index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing source unit sequence, source street sequence, and target sequence.
        """
        unit_position_dataset = torch.tensor(self.unit_position_datasets[index], dtype=torch.float32)
        street_index_sequence = torch.tensor(self.street_index_sequences[index], dtype=torch.long)
        cur_n_street = torch.tensor(self.cur_n_streets[index], dtype=torch.long)

        edge_index_sequence = torch.tensor(self.edge_index_sequences[index], dtype=torch.long)
        adj_matrix = to_dense_adj(edge_index_sequence)[0].numpy()

        n_street = 70
        n_building = 120
        n_boundary = 250
        d_unit = 8
        d_street = 64

        # 패딩된 street position 생성
        street_pos = self.street_unit_position_datasets[index]
        street_indices = self.street_index_sequences[index]
        remove_street_indices = np.array([n_street + 1, n_street + 2, 0])
        street_indices = street_indices[~np.isin(street_indices, remove_street_indices)]
        street_pos = [street_pos[element] for element in street_indices]
        zeros = np.zeros((n_boundary, d_street, 2))
        zeros[0] = 2
        zeros[1:len(street_pos) + 1] = street_pos
        zeros[len(street_pos) + 1] = 3
        zeros[len(street_pos) + 2:] = 4
        street_position_dataset = torch.tensor(zeros, dtype=torch.float32)

        # 패딩된 인접 행렬 생성
        zeros = np.zeros((n_street + n_building, n_street + n_building))
        zeros[0] = 2
        zeros[1:adj_matrix.shape[0] + 1, 1:adj_matrix.shape[1] + 1] = adj_matrix
        zeros[adj_matrix.shape[0] + 1] = 3
        zeros[adj_matrix.shape[0] + 2:] = 4
        adj_matrix_sequence = torch.tensor(zeros, dtype=torch.float32)

        return unit_position_dataset, street_position_dataset, street_index_sequence, adj_matrix_sequence, cur_n_street

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
        - int: Total number of items in the dataset.
        """
        return len(self.unit_position_datasets)