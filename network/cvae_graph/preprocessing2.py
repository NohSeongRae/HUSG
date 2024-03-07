import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import random
import shutil

def preprocesing_dataset(condition_type='graph'):
    save_path = './network/cvae_graph/' + condition_type + '_condition_train_datasets/'

    gpickle_files = [f for f in os.listdir(save_path) if f.endswith('.gpickle')]
    for gpickle_file in tqdm(gpickle_files):
        os.rename(os.path.join(save_path, gpickle_file), os.path.join(save_path, gpickle_file.replace(".geojson", "").replace("boundaries", "buildings")))
    gpickle_files = [f for f in os.listdir(save_path) if f.endswith('.gpickle')]
    random.shuffle(gpickle_files)

    pkl_files = [f for f in os.listdir(save_path) if f.endswith('.pkl')]
    for pkl_file in tqdm(pkl_files):
        os.rename(os.path.join(save_path, pkl_file), os.path.join(save_path, pkl_file.replace(".geojson", "").replace("boundaries", "buildings")))
    pkl_files = [f for f in os.listdir(save_path) if f.endswith('.pkl')]

    print(gpickle_files[0])

    train_len = len(gpickle_files) * 0.8
    val_len = len(gpickle_files) * 0.9
    train_split = gpickle_files[:int(train_len)]
    val_split = gpickle_files[int(train_len):int(val_len)]
    test_split = gpickle_files[int(val_len):]

    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(save_path, folder), exist_ok=True)

    for file_set in [train_split, val_split, test_split]:
        for gpickle_file in tqdm(file_set):
            base_filename = os.path.splitext(gpickle_file)[0]
            pkl_file = base_filename + '.pkl'

            target_folder = 'train' if gpickle_file in train_split else 'val' if gpickle_file in val_split else 'test'
            shutil.move(os.path.join(save_path, gpickle_file), os.path.join(save_path, target_folder, gpickle_file))
            shutil.move(os.path.join(save_path, pkl_file), os.path.join(save_path, target_folder, pkl_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--condition_type", type=str, default="graph", help="Random seed for reproducibility across runs.")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")

    preprocesing_dataset(condition_type=opt.condition_type)