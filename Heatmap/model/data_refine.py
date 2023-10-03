import torch
from PIL import Image
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import h5py

def load_mask_single(args):
    index, image_path =args
    mask_image = Image.open(image_path)
    mask_numpy = np.array(mask_image, dtype=np.float32) * (1.0 / 255.0)
    return index, mask_numpy

# def load_mask_from_hdf5(hdf5_path, start_idx=0, end_idx=None):
#     with h5py.File(hdf5_path, 'r') as hf:
#         if end_idx is None:
#             end_idx=hf["images"].shape[0]
#         masks=hf["images"][start_idx:end_idx]
#         return [np.array(mask, dtype=np.float32) for mask in masks]

def load_mask(dir_name, num_workers=8):
    file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    mask_list=[None]*len(file_list)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_mask_single, enumerate(file_list)))

    # Sort the results based on the index to ensure the original order
    sorted_results = sorted(results, key=lambda x: x[0])
    mask_arrays = [result[1] for result in sorted_results]
    mask_tensors = [torch.tensor(array) for array in mask_arrays]
    return mask_tensors
# def load_mask(hdf5_path):
#     return load_mask_from_hdf5(hdf5_path)
