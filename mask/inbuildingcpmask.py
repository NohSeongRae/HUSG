import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
from tqdm import tqdm
import numpy as np
import imageio
from shapely.geometry import box
from tqdm import tqdm

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)


def get_square_bounds(polygon, padding_percentage=10):
    # building data 전체를 geodataframe형태로 저장
    # gdf = gpd.read_file(geojson_path)

    # 그 전체 data를 감싸는 boundary 찾기
    bounds = polygon.bounds
    # data를 감싸는 사각형의 가로 세로
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    # 정사각형 만들기
    # square_size = max(width, height)
    square_size = max(width, height) * (1 + padding_percentage / 100)

    # 중심좌표 반환
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    # width, height 중 더 값이 큰 것을 한 변의 길이로 하는 정사각형 생성
    square_coords = [
        (center_x - square_size / 2, center_y - square_size / 2),
        (center_x - square_size / 2, center_y + square_size / 2),
        (center_x + square_size / 2, center_y + square_size / 2),
        (center_x + square_size / 2, center_y - square_size / 2),
        (center_x - square_size / 2, center_y - square_size / 2)
    ]

    # left, upper, right, lower 값 추출
    left = square_coords[0][0]
    upper = square_coords[0][1]
    right = square_coords[2][0]
    lower = square_coords[2][1]

    return left, upper, right, lower


def create_masks_for_dataset(center_positions, img_size):
    transform = rasterio.transform.from_origin(0, img_size, 1, 1)

    combined_polygons = []
    masks = []
    for center in center_positions:
        if np.any(center):  # Check if the center is not [0, 0]
            x, y = (center * img_size).astype(int)

            # Create a 3x3 square polygon around the center
            square = box(x - 1.5, y - 1.5, x + 1.5, y + 1.5)
            combined_polygons.append(square)

            # Generate the mask for the current set of polygons
            mask = ~geometry_mask(combined_polygons, transform=transform, invert=False, out_shape=(img_size, img_size))
            masks.append(mask.astype(np.uint8) * 255)

    return masks

def inbuildingcpmask(city_name, image_size, building_center_position_datasets, node_size):
    for idx, dataset in enumerate(tqdm(building_center_position_datasets)):
        # Filter out (0,0) coordinates
        valid_coords = dataset[~np.all(dataset == 0, axis=1)]

        # Create a folder for this dataset
        folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', city_name, 'inbuildigcpmask',
                                   f'{city_name}_{idx + 1}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generate masks based on the number of valid coordinates
        for i in range(1, len(valid_coords) + 1):
            polygons = []
            for coord in valid_coords[:i]:
                # Create a small box around the coordinate (3x3 pixels)
                minx = coord[0] - (node_size/2) / image_size
                miny = coord[1] - (node_size/2) / image_size
                maxx = coord[0] + (node_size/2) / image_size
                maxy = coord[1] + (node_size/2) / image_size
                polygons.append(box(minx, miny, maxx, maxy))

            # Use rasterio to create the mask
            transform = rasterio.transform.from_bounds(0, 0, 1, 1, image_size, image_size)
            mask = geometry_mask(polygons, transform=transform, invert=True, out_shape=(image_size, image_size))

            # Save the mask to the appropriate folder
            mask_path = os.path.join(folder_path, f'{city_name}_{idx + 1}_{i}.png')
            mask = (mask * 1).astype(np.uint8)  # Convert to values between 0 and 255
            imageio.imsave(mask_path, mask)
