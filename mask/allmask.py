import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from skimage.morphology import dilation, square
from shapely.geometry import Polygon, LineString
from tqdm import tqdm
import numpy as np
import imageio
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


def allmask(city_name, image_size, unit_coords_datasets, street_index_sequences, building_index_sequences, linewidth=1):
    invalid_indices = []

    width, height = image_size, image_size

    for dataset_idx in tqdm(range(len(unit_coords_datasets))):
        final_mask = np.zeros((height, width), dtype=np.uint8)
        segment_index = 0
        unit_coords_dataset = unit_coords_datasets[dataset_idx][
            np.any(unit_coords_datasets[dataset_idx] != 0, axis=(1, 2))]

        # Padding 값을 무시
        unit_coords_dataset = unit_coords_dataset[unit_coords_dataset[:, 0, 0] != 0]

        coordinates = [segment[0] for segment in unit_coords_dataset]
        coordinates.append(unit_coords_dataset[-1][1])

        boundary_polygon = Polygon(coordinates)

        for segment in unit_coords_dataset:
            line = LineString(segment)
            boundaries_list = [line]

            # 이미지 경계 설정
            left, bottom, right, top = get_square_bounds(boundary_polygon)
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

            try:
                boundary_mask = geometry_mask(boundaries_list, transform=transform, invert=True, out_shape=(height, width))
            except Exception as e:
                print(f"No valid geometry objects {dataset_idx + 1} : {e}")
                invalid_indices.append(dataset_idx + 1)
                continue

            try:
                inside_mask = geometry_mask([boundary_polygon], transform=transform, invert=True, out_shape=(height, width))
            except Exception as e:
                print(f"No valid geometry objects {dataset_idx + 1} : {e}")
                invalid_indices.append(dataset_idx + 1)
                continue

            thick_boundary_mask = dilation(boundary_mask, square(3))
            # thick_boundary_mask = dilation(boundary_mask, square(1))

            # street_index_sequences에서 padding 값을 무시
            if street_index_sequences[dataset_idx][segment_index] != 0:
                street_idx = int(street_index_sequences[dataset_idx][segment_index])
                final_mask[thick_boundary_mask] = street_idx + 1

                final_mask[inside_mask == 1] = 1

            # building_index_sequences padding 값을 무시
            if building_index_sequences[dataset_idx][segment_index] != 2:
                tf_output = int(building_index_sequences[dataset_idx][segment_index])
                if tf_output == 0:
                    filtered_values = [val for val in street_index_sequences[dataset_idx] if val < 49]
                    # filtered_values = street_index_sequences[dataset_idx][street_index_sequences[dataset_idx] < 49]
                    final_mask[thick_boundary_mask] = np.max(filtered_values) + 1

            # final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)

            segment_index += 1

        final_mask = final_mask.astype(np.uint8)
        #
        # # 마스크 저장
        # allmask_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', f'{city_name}',
        #                                        'allmask')
        #
        # if not os.path.exists(allmask_folderpath):
        #     os.makedirs(allmask_folderpath)
        #
        # allmask_filename = os.path.join(allmask_folderpath, f'{city_name}_{dataset_idx + 1}.png')
        # imageio.imsave(allmask_filename, final_mask)

        mask_coords = {}
        unique_values = np.unique(final_mask)
        for value in unique_values:
            y_positions, x_positions = np.where(final_mask == value)
            coords_list = list(zip(y_positions, x_positions))
            mask_coords[value] = coords_list

        # 해당 데이터셋에 대한 좌표를 pickle 파일로 저장
        save_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', 'mask_pickle', f'{city_name}',
                                       'allmask')

        if not os.path.exists(save_folderpath):
            os.makedirs(save_folderpath)
        pickle_path = os.path.join(save_folderpath, f'{city_name}_{dataset_idx + 1}.pkl')

        with open(pickle_path, "wb") as f:
            pickle.dump(mask_coords, f)

    if len(invalid_indices) > 0:
        invalid_indices_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask', "mask_pickle",
                                            city_name, f"{city_name}_all_invalid.pkl")

        with open(invalid_indices_path, 'wb') as f:
            pickle.dump(invalid_indices, f)