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

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)


def get_square_bounds(geojson_path, padding_percentage=10):
    # building data 전체를 geodataframe형태로 저장
    gdf = gpd.read_file(geojson_path)

    # 그 전체 data를 감싸는 boundary 찾기
    bounds = gdf.total_bounds
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


def buildingmask(city_name, image_size):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    for i in tqdm(range(1, filenum + 1)):
        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'density20_building120_Normalized',
                                         'Boundaries', f'{city_name}_boundaries{i}.geojson')

        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'density20_building120_Normalized',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')

        if os.path.exists(building_filename):
            buildings_gdf = gpd.read_file(building_filename)
            width, height = image_size, image_size
            left, bottom, right, top = get_square_bounds(boundary_filename)
            transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

            final_mask = np.ones((height, width), dtype=np.uint8) * 0

            inside_building_mask = geometry_mask(buildings_gdf.geometry, transform=transform, invert=True,
                                                 out_shape=(height, width))
            final_mask[inside_building_mask] = 255

            buildingmask_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '1_evaluation',
                                                 'layoutvae', 'allbuildingmask', f'{city_name}_buildingmask{i}.png')

            imageio.imsave(buildingmask_filename, final_mask)

if __name__=='__main__':
    city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
    "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
    "sanfrancisco", "miami", "seattle", "boston", "providence",
    "neworleans", "denver", "pittsburgh", "tampa", "washington"]

    for city_name in city_names:
        buildingmask(city_name, 256)