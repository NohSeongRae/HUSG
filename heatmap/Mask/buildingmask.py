import cv2
import numpy as np
import os

city_name = "philadelphia"


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.where(gray == 255, 255, 0)
    non_white_area = np.where(gray != 255, 0, 255)


    result = cv2.bitwise_or(mask, non_white_area)

    return result


dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
num_file = len(files)

for i in range(1, num_file + 1):
    image_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'BuildingImage', f'{city_name}_boundaries_image{i}.png')

    if os.path.exists(image_filepath):
        image_path = image_filepath
        polygon_mask = process_image(image_filepath)

        save_path_polygon = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'buildingmask', f'{city_name}_polygonmask{i}.png')

        cv2.imwrite(save_path_polygon, polygon_mask)