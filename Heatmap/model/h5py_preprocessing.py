import h5py
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import paths


def images_to_hdf5(img_dir, hdf5_path, img_size):
    file_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('images', (len(file_list), img_size, img_size), dtype=np.float32)

        for i, image_path in tqdm(enumerate(file_list), total=len(file_list), desc="Saving to HDF5"):
            image = Image.open(image_path)
            image_np = np.array(image, dtype=np.float32) * (1.0 / 255.0)
            hf["images"][i, ...] = image_np


images_to_hdf5(paths.boundarymask_all, paths.hdf5_boundarymask,224)
# images_to_hdf5(paths.insidemask_all, paths.hdf5_insidemask,224)
# images_to_hdf5(paths.centroidmask_all, paths.hdf5_centroidmask,64)
