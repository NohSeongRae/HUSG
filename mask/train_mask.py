import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

### street mask

city_name = "atlanta"

### allmask, streetmask

def streetmask(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'streetmask', 'atlanta_1.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    plt.imshow(image, cmap='gray')
    plt.show()

def allmask(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'allmask', 'atlanta_1.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    plt.imshow(image, cmap='gray')
    plt.show()


def allmask_num(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'allmask', f'{city_name}_1.pkl')  # Modified filename

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    for i in range(120):
        for j in range(120):
            if image[i, j] != 0:  # Only display non-zero values
                plt.text(j, i, str(image[i, j]), ha='center', va='center', color='red', fontsize=6)

    plt.axis('off')
    plt.show()

# insidemask

def insidemask(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'allmask', 'atlanta_1.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    inside_mask = (image == 1).astype(np.uint8)

    plt.imshow(inside_mask, cmap='gray')
    plt.show()

def boundarymask(city_name): ## 다시 (boundary만 따로 저장)
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'allmask', 'atlanta_1.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    boundary_mask = (image > 1).astype(np.uint8)

    plt.imshow(boundary_mask, cmap='gray')
    plt.show()

def inbuildingcpmask(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'inbuildingcpmask', 'atlanta_1', 'atlanta_1_3.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    plt.imshow(image, cmap='gray')
    plt.show()

def tfoutputseqmask(city_name): ## 수정?
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'tfoutput_seqmask', 'atlanta_1.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    mask = np.zeros((120, 120), dtype=np.uint8)

    for y, x in mask_coords:
        mask[y, x] = 1

    plt.imshow(mask, cmap='gray')
    plt.show()

def tfoutputplotmask(city_name): ## 수정?
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'tfoutput_plotmask', 'atlanta_2.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    mask = np.zeros((120, 120), dtype=np.uint8)

    for y, x in mask_coords:
        mask[y, x] = 1

    plt.imshow(mask, cmap='gray')
    plt.show()

def inedgemask(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'inedgemask', 'atlanta_1', 'atlanta_1_2.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    plt.imshow(image, cmap='gray')
    plt.show()

def groundtruthmask(city_name):
    save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle',
                                   f'{city_name}', 'groundtruthmask', 'atlanta_1', 'atlanta_1_1.pkl')

    with open(save_folderpath, 'rb') as f:
        mask_coords = pickle.load(f)

    image = np.zeros((120, 120), dtype=np.uint8)

    for pixel_value, coords in mask_coords.items():
        for y, x in coords:
            image[y, x] = pixel_value

    plt.imshow(image, cmap='gray')
    plt.show()

# allmask(city_name) ## allmask 수정

groundtruthmask(city_name)