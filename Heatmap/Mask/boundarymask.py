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




def get_square_bounds(geojson_path):
    # building data 전체를 geodataframe형태로 저장
    gdf = gpd.read_file(geojson_path)

    # 그 전체 data를 감싸는 boundary 찾기
    bounds = gdf.total_bounds
    # data를 감싸는 사각형의 가로 세로
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    # 정사각형 만들기
    square_size = max(width, height)
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


def boundarymask(city_name):
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

            width, height = 224, 224
            # 이미지 경계 설정
            left, bottom, right, top = get_square_bounds(boundary_filename)
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

            boundary_mask = geometry_mask(boundaries_list, transform=transform, invert=True, out_shape=(height, width))

            # 경계선 굵기 조절
            thick_boundary_mask = dilation(boundary_mask, square(1))

            inverted_mask = np.where(thick_boundary_mask, 0, 255).astype(np.uint8)

            boundarymask_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'mask',
                                             'boundarymask', f'{city_name}_boundarymask{i}.png')

            imageio.imsave(boundarymask_filename, inverted_mask)
