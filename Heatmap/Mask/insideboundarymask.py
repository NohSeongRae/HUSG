import cv2
import numpy as np
import os

city_name = "atlanta"

def create_masks(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, boundary_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    boundary_mask = 1 - (boundary_binary / 255)

    filled = boundary_binary.copy()
    mask = np.zeros((boundary_binary.shape[0] + 2, boundary_binary.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, mask, (0, 0), 255)

    # 경계선 외부를 0로 설정, 내부를 1으로 설정 (색 반전)
    inside_mask = np.where(filled == 255, 1, 0)

    return boundary_mask, inside_mask


dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
num_file = len(files)

for i in range(1, num_file + 1):
    image_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'BoundaryImage', f'{city_name}_boundaries_image{i}.png')

    if os.path.exists(image_filepath):
        image_path = image_filepath
        boundary_mask, inside_mask = create_masks(image_path)

        save_path_boundary = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'boundarymask', f'{city_name}_boundarymask{i}.png')
        save_path_inside = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'insidemask', f'{city_name}_boundarymask{i}.png')

        cv2.imwrite(save_path_boundary, boundary_mask * 255)
        cv2.imwrite(save_path_inside, inside_mask * 255)