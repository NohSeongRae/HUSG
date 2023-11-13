import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import networkx as nx
import random
import shutil
from torch_geometric.utils import dense_to_sparse, to_dense_adj

def preprocesing_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                      n_street=60, n_building=120, n_boundary=200, d_unit=8, d_street=64, condition_type='graph'):

    train_split_path = './network/cvae_graph/whole_city/train_split.pkl'
    val_split_path = './network/cvae_graph/whole_city/val_split.pkl'
    test_split_path = './network/cvae_graph/whole_city/test_split.pkl'

    with open(train_split_path, 'rb') as f:
        train_split = pickle.load(f)
        train_split = [s.replace("geojson", "gpickle") for s in train_split]
        train_split.sort()
    with open(val_split_path, 'rb') as f:
        val_split = pickle.load(f)
        val_split = [s.replace("geojson", "gpickle") for s in val_split]
        val_split.sort()
    with open(test_split_path, 'rb') as f:
        test_split = pickle.load(f)
        test_split = [s.replace("geojson", "gpickle") for s in test_split]
        test_split.sort()

    save_path = './network/cvae_graph/' + condition_type + '_condition_train_datasets/'

    gpickle_files = [f for f in os.listdir(save_path) if f.endswith('.gpickle')]
    gpickle_files = [s.replace(".geojson", "") for s in gpickle_files]
    gpickle_files = [s.replace("boundaries", "buildings") for s in gpickle_files]
    gpickle_files.sort()

    train_result = all(elem in gpickle_files for elem in train_split)
    val_result = all(elem in gpickle_files for elem in val_split)
    test_result = all(elem in gpickle_files for elem in test_split)

    print(np.array(train_split), np.array(train_split).shape)
    print(np.array(val_split), np.array(val_split).shape)
    print(np.array(test_split), np.array(test_split).shape)

    print(np.array(gpickle_files))

    print(train_result)
    print(val_result)
    print(test_result)

    # # 분할 지점 계산
    # total_files = len(gpickle_files)
    # train_end = int(total_files * train_ratio)
    # val_end = train_end + int(total_files * val_ratio)
    # print(total_files, train_end, val_end)
    #
    # # 각 데이터셋에 대한 파일 목록
    # train_files = gpickle_files[:train_end]
    # val_files = gpickle_files[train_end:val_end]
    # test_files = gpickle_files[val_end:]
    #
    # # 폴더 생성 (존재하지 않을 경우)
    # for folder in ['train', 'val', 'test']:
    #     os.makedirs(os.path.join(save_path, folder), exist_ok=True)
    #
    # # 파일 묶음을 해당 폴더로 이동
    # for file_set in [train_files, val_files, test_files]:
    #     for gpickle_file in tqdm(file_set):
    #         base_filename = os.path.splitext(gpickle_file)[0]
    #         pkl_file = base_filename + '.pkl'
    #
    #         target_folder = 'train' if gpickle_file in train_files else 'val' if gpickle_file in val_files else 'test'
    #         shutil.move(os.path.join(save_path, gpickle_file),
    #                     os.path.join(save_path, target_folder, gpickle_file))
    #         shutil.move(os.path.join(save_path, pkl_file), os.path.join(save_path, target_folder, pkl_file))


if __name__ == '__main__':
    # Set the argparse
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    # Define the arguments with their descriptions
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--d_street", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--d_unit", type=int, default=8, help="Dimension of the model.")
    parser.add_argument("--n_building", type=int, default=120, help="binary classification for building existence.")
    parser.add_argument("--n_boundary", type=int, default=250, help="Number of boundary or token.")
    parser.add_argument("--n_street", type=int, default=60, help="Number of boundary or token.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--condition_type", type=str, default="graph", help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Convert namespace to dictionary and iterate over it to print all key-value pairs
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(train_ratio=opt.train_ratio, val_ratio=opt.val_ratio, test_ratio=opt.test_ratio,
                         n_street=opt.n_street, n_building=opt.n_building,
                         n_boundary=opt.n_boundary, d_unit=opt.d_unit, d_street=opt.d_street,
                         condition_type=opt.condition_type)