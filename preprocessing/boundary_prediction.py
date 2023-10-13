import pickle
import numpy as np

unit_coords_path = './dataset/husg_unit_coords.pkl'
npz_path = './dataset/husg_transformer_dataset.npz'

with open(unit_coords_path, 'rb') as f:
    unit_coords_data = pickle.load(f)

npz_data = np.load(npz_path)

print(npz_data['building_index_sequences'])

# print(unit_coords_data)

