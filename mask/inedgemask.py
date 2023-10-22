import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, LineString, Point
from skimage.morphology import dilation, square
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

def inedgemask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, node_size):
    for idx, dataset in enumerate(tqdm(building_center_position_datasets)):
        unit_coords_dataset = unit_coords_datasets[idx][np.any(unit_coords_datasets[idx] != 0, axis=(1, 2))]

        coordinates = [segment[0] for segment in unit_coords_dataset]
        coordinates.append(unit_coords_dataset[-1][1])

        boundary_polygon = Polygon(coordinates)

        # Create a folder for this dataset
        folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', city_name, 'inedgemask',
                                   f'{city_name}_{idx + 1}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Filter out (0,0) coordinates
        valid_coords = dataset[~np.all(dataset == 0, axis=1)]

        # 1. Use the get_square_bounds function to get the bounding coordinates
        left, upper, right, lower = get_square_bounds(boundary_polygon)

        # 2. Set the transform using these coordinates
        width, height = image_size, image_size
        transform = rasterio.transform.from_bounds(left, upper, right, lower, width, height)

        line_mask = np.zeros((image_size, image_size), dtype=np.uint8)

        for i, coord in enumerate(valid_coords):
            lines = []  # Initialize the lines list at the beginning of each loop iteration

            # 1. Create a line perpendicular to the nearest point on the boundary polygon
            nearest_point = boundary_polygon.boundary.interpolate(boundary_polygon.boundary.project(Point(coord)))
            vertical_line = LineString([coord, (nearest_point.x, nearest_point.y)])
            lines.append(vertical_line)

            # 2. Create a line between the current building and the previous one
            if i > 0:
                connection_line = LineString([valid_coords[i - 1], coord])
                lines.append(connection_line)

            # 3. Generate the mask using rasterio for lines
            current_edge_mask = geometry_mask(lines, transform=transform, invert=True,
                                              out_shape=(image_size, image_size))
            current_edge_mask = dilation(current_edge_mask, square(5))

            # Update the current edge to have a value of 2 in line_mask
            line_mask[current_edge_mask == 1] = 2

            polygons = []
            for coord in valid_coords[:i + 1]:
                # Create a small box around the coordinate (3x3 pixels)
                minx = coord[0] - (node_size / 2) / image_size
                miny = coord[1] - (node_size / 2) / image_size
                maxx = coord[0] + (node_size / 2) / image_size
                maxy = coord[1] + (node_size / 2) / image_size
                polygons.append(box(minx, miny, maxx, maxy))

            building_mask = geometry_mask(polygons, transform=transform, invert=True,
                                          out_shape=(image_size, image_size))

            line_mask[building_mask == 1] = 0

            # Save the mask to the appropriate folder
            mask_path = os.path.join(folder_path, f'{city_name}_{idx + 1}_{i + 1}.png')
            combined_mask = (line_mask * 1).astype(np.uint8)  # Convert to values between 0 and 255
            imageio.imsave(mask_path, combined_mask)

            line_mask[line_mask == 2] = 1


if __name__=="__main__":
    city_names = ["neworleans"]

    image_size = 120
    linewidth = 5
    num_grids = 60
    unit_length = 0.04
    cp_node_size = 5

    for city_name in city_names:
        print("city name", city_name)
        city_npz_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer',
                                     'train_dataset', f'{city_name}', 'husg_transformer_dataset.npz')
        building_center_npz_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer',
                                                'train_dataset', f'{city_name}', 'husg_building_center_position.npz')

        city_npz_data = np.load(city_npz_path)
        building_npz_data = np.load(building_center_npz_path)

        building_index_sequences = city_npz_data['building_index_sequences']
        street_index_sequences = city_npz_data['street_index_sequences']
        # unit_position_datasets = city_npz_data['unit_position_datasets']
        # street_unit_position_datasets = city_npz_data['street_unit_position_datasets']
        unit_coords_datasets = city_npz_data['unit_coords_datasets']

        building_center_position_datasets = building_npz_data['building_center_position_datasets']

        inedgemask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, cp_node_size)