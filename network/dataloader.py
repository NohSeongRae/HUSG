import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle

class GraphDataset(Dataset):
    def __init__(self, data_type='train'):
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
        """주어진 행렬을 pad_shape 크기로 패딩하고, 패딩 마스크도 함께 반환합니다."""
        original_shape = matrix.shape
        pad_width = ((0, max(0, pad_shape[0] - original_shape[0])),
                     (0, max(0, pad_shape[1] - original_shape[1])))
        padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)

        # 패딩 마스크 생성: 실제 데이터는 0, 패딩 부분은 1
        pad_mask = np.zeros_like(padded_matrix, dtype=np.float32)
        pad_mask[:original_shape[0], :original_shape[1]] = 1  # 실제 데이터 부분을 1로 설정

        return padded_matrix, pad_mask

    def __getitem__(self, idx):
        load_path = self.folder_path + '/' + self.pkl_files[idx]
        with open(load_path, 'rb') as f:
            self.data = pickle.load(f)

        boundary_adj_matrix = self.data['boundary_adj_matrix']
        building_adj_matrix = self.data['building_adj_matrix']
        bb_adj_matrix = self.data['bb_adj_matrix']
        boundary_pos_feature = self.data['boundary_pos_feature']
        n_boundary = self.data['n_boundary']
        n_building = self.data['n_building']

        # 각 행렬을 원하는 크기로 패딩
        boundary_adj_matrix_padded, boundary_pad_mask = self.pad_matrix(boundary_adj_matrix, (200, 200))
        building_adj_matrix_padded, building_pad_mask = self.pad_matrix(building_adj_matrix, (120, 120))
        bb_adj_matrix_padded, bb_pad_mask = self.pad_matrix(bb_adj_matrix, (120, 200))
        boundary_pos_padded, _ = self.pad_matrix(boundary_pos_feature, (120, 2))

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
        return self.data_length