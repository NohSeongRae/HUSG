import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from skimage.morphology import dilation, disk, square
from tqdm import tqdm
import numpy as np
import imageio
from shapely.geometry import shape, MultiPolygon

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


def buildingcentroid(city_name, image_size):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    for i in tqdm(range(1, filenum + 1)):
        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'filtered_data',
                                         'Boundaries', f'{city_name}_boundaries{i}.geojson')

        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'filtered_data',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')

        centroids = []
        if os.path.exists(building_filename):
            with open(building_filename, 'r') as file:
                data = json.load(file)

            for feature in data['features']:
                building_geometry = feature['geometry']
                building_polygon = shape(building_geometry)
                centroids.append(building_polygon.centroid.coords[0])

            width, height = image_size, image_size

            building_mask = np.ones((height, width), dtype=np.uint8) * 255
            left, bottom, right, top = get_square_bounds(boundary_filename)
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

            for centroid in centroids:
                row, col = rasterio.transform.rowcol(transform, centroid[0], centroid[1])
                building_mask[row, col] = 0

            buildingmask_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '0_others', 'mask_512', 'centroidmask',
                                                 f'{city_name}_centroidmask{i}.png')

            building_mask = 255 - building_mask

            imageio.imsave(buildingmask_filename, building_mask)
