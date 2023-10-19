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



def insidemask(city_name, image_size, unit_coords_datasets):
    for idx in tqdm(range(len(unit_coords_datasets))):
        unit_coords_dataset = unit_coords_datasets[idx][np.any(unit_coords_datasets[idx] != 0, axis=(1, 2))]

        coordinates = [segment[0] for segment in unit_coords_dataset]
        coordinates.append(unit_coords_dataset[-1][1])

        boundary_polygon = Polygon(coordinates)

        boundary_line = boundary_polygon.boundary
        boundaries_list = [boundary_line]

        width, height = image_size, image_size
        # 이미지 경계 설정
        left, bottom, right, top = get_square_bounds(boundary_polygon)
        transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

        mask = geometry_mask([boundary_polygon], transform=transform, invert=True, out_shape=(height, width))

        scaled_mask = (mask * 1).astype(np.uint8)

        # scaled_mask = 255 - scaled_mask

        insidemask_folderpath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '3_mask',
                                             f'{city_name}', 'insidemask')

        if not os.path.exists(insidemask_folderpath):
            os.makedirs(insidemask_folderpath)

        insidemask_filename = os.path.join(insidemask_folderpath, f'{city_name}_{idx+1}.png')

        imageio.imsave(insidemask_filename, scaled_mask)