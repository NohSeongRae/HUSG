import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from skimage.morphology import dilation, square
from tqdm import tqdm
import numpy as np
import imageio

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)


def get_square_bounds(geojson_path, padding_ratio=0.05):
    gdf = gpd.read_file(geojson_path)
    bounds = gdf.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    square_size = max(width, height)
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    padding = square_size * padding_ratio
    square_size += 2 * padding

    square_coords = [
        (center_x - square_size / 2, center_y - square_size / 2),
        (center_x + square_size / 2, center_y - square_size / 2),
        (center_x + square_size / 2, center_y + square_size / 2),
        (center_x - square_size / 2, center_y + square_size / 2)
    ]

    return square_coords[0][0], square_coords[1][1], square_coords[2][0], square_coords[3][1]


def boundarymask(city_name, image_size):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    for i in tqdm(range(1, filenum + 1)):
        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'filtered_data',
                                         'Boundaries', f'{city_name}_boundaries{i}.geojson')

        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'filtered_data',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')

        if os.path.exists(building_filename):
            gdf = gpd.read_file(boundary_filename)
            boundaries = gdf.geometry.boundary

            boundaries_list = list(boundaries)

            width, height = image_size, image_size
            # 이미지 경계 설정
            left, bottom, right, top = get_square_bounds(boundary_filename)
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

            boundary_mask = geometry_mask(boundaries_list, transform=transform, invert=True, out_shape=(height, width))

            # 경계선 굵기 조절
            thick_boundary_mask = dilation(boundary_mask, square(5))

            inverted_mask = np.where(thick_boundary_mask, 0, 255).astype(np.uint8)

            inverted_mask = 255 - inverted_mask

            boundarymask_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', f'mask_{image_size}',
                                             'boundarymask', f'{city_name}_boundarymask{i}.png')

            imageio.imsave(boundarymask_filename, inverted_mask)