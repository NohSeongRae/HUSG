import torch
from PIL import Image
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def load_mask_single(image_path):
    mask_image=Image.open(image_path)
    mask_numpy=np.array(mask_image, dtype=np.float32)*(1.0/255.0)
    return torch.tensor(mask_numpy)
def load_mask(dir_name, num_workers=8):
    mask_list = []

    file_list = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    mask_list=[None]*len(file_list)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        mask_tensors=list(executor.map(load_mask_single, file_list))
    # for i in range(len(file_list)-1):
    #     image_path_png = file_list[i]
    #     image_path = os.path.join(dir_name, image_path_png)
    #     mask_image = Image.open(image_path)
    #     mask_numpy = np.array(mask_image, dtype=np.float32) / 255.0
    #     mask_tensor = torch.tensor(mask_numpy)
    #     mask_list.append(mask_tensor)

    return mask_tensors

