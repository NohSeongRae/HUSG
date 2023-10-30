import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
from torch_geometric.utils import dense_to_sparse, to_dense_adj

def preprocesing_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                      n_street=60, n_building=120, n_boundary=200, d_unit=8, d_street=64):
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
                        zeros[1:len(data) + 1] = data
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
    full_dataset = {
        'unit_position_datasets': np.array(all_unit_position_datasets),
        'street_unit_position_datasets': np.array(all_street_unit_position_datasets),
        'street_index_sequences': np.array(all_street_index_sequences),
        'edge_index_sequences': np.array(all_edge_index_sequences),
        'cur_n_streets': np.array(all_cur_n_street)
    }

    # Shuffle the dataset
    permuted_indices = np.random.permutation(len(full_dataset['unit_position_datasets']))
    for key in full_dataset:
        full_dataset[key] = full_dataset[key][permuted_indices]

    total_size = len(full_dataset['unit_position_datasets'])

    for data_type in ['train', 'val', 'test']:
        if data_type == 'train':
            start_index = 0
            end_index = int(total_size * train_ratio)
        elif data_type == 'val':
            start_index = int(total_size * train_ratio)
            end_index = int(total_size * (train_ratio + val_ratio))
        else:
            start_index = int(total_size * (train_ratio + val_ratio))
            end_index = int(total_size * (train_ratio + val_ratio + test_ratio))

        save_path = './network/transformer_graph/' + data_type + '_datasets'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(save_path,
                 unit_position_datasets=full_dataset['unit_position_datasets'][start_index:end_index],
                 street_unit_position_datasets=full_dataset['street_unit_position_datasets'][start_index:end_index],
                 street_index_sequences=full_dataset['street_index_sequences'][start_index:end_index],
                 edge_index_sequences=full_dataset['edge_index_sequences'][start_index:end_index],  # 수정된 부분
                 cur_n_streets=full_dataset['cur_n_streets'][start_index:end_index])

if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--train_ratio", type=float, default=0.89, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--d_street", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--d_unit", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--n_building", type=int, default=120, help="binary classification for building existence.")
    parser.add_argument("--n_boundary", type=int, default=250, help="Number of boundary or token.")
    parser.add_argument("--n_street", type=int, default=60, help="Number of boundary or token.")

    opt = parser.parse_args()

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
                         n_street=opt.n_street, n_building=opt.n_building,
                         n_boundary=opt.n_boundary, d_unit=opt.d_unit, d_street=opt.d_street)