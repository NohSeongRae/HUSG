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
import pickle

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

def groundtruthmask(city_name, image_size, unit_coords_datasets, building_center_position_datasets):
    invalid_indices = []

    for idx, dataset in enumerate(tqdm(building_center_position_datasets)):
        unit_coords_dataset = unit_coords_datasets[idx][np.any(unit_coords_datasets[idx] != 0, axis=(1, 2))]

        coordinates = [segment[0] for segment in unit_coords_dataset]
        coordinates.append(unit_coords_dataset[-1][1])

        boundary_polygon = Polygon(coordinates)

        # Create a folder for this dataset
        folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', city_name, 'groundtruthmask',
                                   f'{city_name}_{idx + 1}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        rows_with_1 = dataset[dataset[:, 0] == 1]
        valid_coords = rows_with_1[:, [1, 2]]

        left, upper, right, lower = get_square_bounds(boundary_polygon)

        # 2. Set the transform using these coordinates
        width, height = image_size, image_size
        transform = rasterio.transform.from_bounds(left, upper, right, lower, width, height)

        for i, coord in enumerate(valid_coords):
            lines = []

            nearest_point = boundary_polygon.boundary.interpolate(boundary_polygon.boundary.project(Point(coord)))
            vertical_line = LineString([coord, (nearest_point.x, nearest_point.y)])
            lines.append(vertical_line)

            if i > 0:
                connection_line = LineString([valid_coords[i - 1], coord])
                lines.append(connection_line)

            try:
                line_mask = geometry_mask(lines, transform=transform, invert=True, out_shape=(image_size, image_size))
            except Exception as e:
                print(f"No valid geometry objects {idx + 1} : {e}")
                invalid_indices.append(idx + 1)
                continue

            current_building_mask = np.zeros((image_size, image_size), dtype=np.uint8)
            current_building_mask[int(coord[1]), int(coord[0])] = 2

            node_size = 1

            minx = coord[0] - (node_size / 2) / image_size
            miny = coord[1] - (node_size / 2) / image_size
            maxx = coord[0] + (node_size / 2) / image_size
            maxy = coord[1] + (node_size / 2) / image_size

            try:
                current_building_mask = geometry_mask([box(minx, miny, maxx, maxy)], transform=transform, invert=True,
                                                      out_shape=(image_size, image_size))
            except Exception as e:
                print(f"No valid geometry objects {idx + 1} : {e}")
                continue

            current_building_mask = current_building_mask * 2

            if i > 0:
                prev_coord = valid_coords[i - 1]
                previous_building_mask = np.zeros((image_size, image_size), dtype=np.uint8)
                previous_building_mask[int(prev_coord[1]), int(prev_coord[0])] = 1
            else:
                previous_building_mask = np.zeros((image_size, image_size), dtype=np.uint8)

            combined_mask = np.where(current_building_mask == 2, 2, np.maximum(line_mask, previous_building_mask))

            # 1과 2의 값을 가진 픽셀의 좌표를 찾습니다.
            y_positions_1, x_positions_1 = np.where(combined_mask == 1)
            y_positions_2, x_positions_2 = np.where(combined_mask == 2)

            coords_1 = list(zip(y_positions_1, x_positions_1))
            coords_2 = list(zip(y_positions_2, x_positions_2))

            coords_dict = {
                1: coords_1,
                2: coords_2
            }

            pickle_folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', "mask_pickle", city_name, 'groundtruthmask',
                                   f'{city_name}_{idx + 1}')

            if not os.path.exists(pickle_folder_path):
                os.makedirs(pickle_folder_path)

            pickle_path = os.path.join(pickle_folder_path, f'{city_name}_{idx + 1}_{i+1}.pkl')

            with open(pickle_path, "wb") as f:
                pickle.dump(coords_dict, f)


    if len(invalid_indices) > 0:
        invalid_indices_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', "mask_pickle",
                                            city_name, f"{city_name}_groundtruth_invalid.pkl")

        with open(invalid_indices_path, 'wb') as f:
            pickle.dump(invalid_indices, f)

if __name__=="__main__":
    import pickle

    city_names = ["atlanta"]

    image_size = 120
    linewidth = 5
    num_grids = 60
    unit_length = 0.04
    cp_node_size = 3
    line_width = 2

    pickle_folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer',
                                      'train_dataset')

    for city_name in city_names:
        print("city name", city_name)
        building_exist_sequences_path = os.path.join(pickle_folder_path, f'{city_name}', 'building_exist_sequences.pkl')
        street_index_sequences_path = os.path.join(pickle_folder_path, f'{city_name}', 'street_index_sequences.pkl')
        unit_coords_datasets_path = os.path.join(pickle_folder_path, f'{city_name}', 'unit_coords_datasets.pkl')
        node_features_path = os.path.join(pickle_folder_path, f'{city_name}', 'node_features.pkl')

        with open(building_exist_sequences_path, 'rb') as f:
            building_exist_sequences = pickle.load(f)

        with open(street_index_sequences_path, 'rb') as f:
            street_index_sequences = pickle.load(f)

        with open(unit_coords_datasets_path, 'rb') as f:
            unit_coords_datasets = pickle.load(f)

        with open(node_features_path, 'rb') as f:
            building_center_position_datasets = pickle.load(f)

        groundtruthmask(city_name, image_size, unit_coords_datasets, building_center_position_datasets, cp_node_size, line_width)