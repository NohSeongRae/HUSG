import torch
import numpy as np
import pickle
import argparse
import os
import random

def preprocesing_dataset(train_ratio=0.8, val_ratio=0.1, condition_type='graph'):
    save_path = './network/cvae_graph/' + condition_type + '_condition_train_datasets/'

    gpickle_files = [f for f in os.listdir(save_path) if f.endswith('.gpickle')]
    for gpickle_file in gpickle_files:
        os.rename(os.path.join(save_path, gpickle_file),
                  os.path.join(save_path, gpickle_file.replace(".geojson", "").replace("boundaries", "buildings")))
    gpickle_files = [f for f in os.listdir(save_path) if f.endswith('.gpickle')]
    random.shuffle(gpickle_files)

    total_files = len(gpickle_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    print(total_files, train_end, val_end)

    train_files = gpickle_files[:train_end]
    val_files = gpickle_files[train_end:val_end]
    test_files = gpickle_files[val_end:val_end + 1000]

    print(np.array(train_files), np.array(train_files).shape)
    print(np.array(val_files), np.array(val_files).shape)
    print(np.array(test_files), np.array(test_files).shape)

    train_split_path = './network/cvae_graph/whole_city/train_split.pkl'
    val_split_path = './network/cvae_graph/whole_city/val_split.pkl'
    test_split_path = './network/cvae_graph/whole_city/test_split.pkl'

    with open(train_split_path, 'wb') as file:
        pickle.dump(train_files, file)
    with open(val_split_path, 'wb') as file:
        pickle.dump(val_files, file)
    with open(test_split_path, 'wb') as file:
        pickle.dump(test_files, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--train_ratio", type=float, default=0.8, help="Use checkpoint index.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Use checkpoint index.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--condition_type", type=str, default="graph",
                        help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(train_ratio=opt.train_ratio, val_ratio=opt.val_ratio,
                         condition_type=opt.condition_type)
