import os
from PIL import Image
import numpy as np
from data_refine import *
import shutil
import paths
def generate_and_test():
    # Create a temporary directory to store dummy images
    os.makedirs("test_dir", exist_ok=True)


    # Test the load_mask function
    masks = load_mask(paths.buildingmask_sample)

    # Verify the order by printing the filenames
    for i, tensor in enumerate(masks):
        # Derive filename from tensor data (this is just a mock method for demonstration purposes)
        print(f"Loaded tensor {i}, derived filename: {i}.png")



if __name__ == '__main__':
    generate_and_test()