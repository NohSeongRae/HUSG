# key property 추가
import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import variables as variables
from etc import filepath as filepath
from cal_area_ratio import getarearatio

# 자를 영역 추출
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


def image(city_name):
    name_list = []
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    index = 0
    filesaveindex = 1

    under10percent = getarearatio(city_name)

    for i in range(1, filenum + 1):
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Combined_Buildings', f'{city_name}_buildings{i}.geojson')

        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Boundaries', f'{city_name}_boundaries{i}.geojson')

        if os.path.exists(building_filename):
            left, upper, right, lower = get_square_bounds(building_filename)

            with open(building_filename, "r", encoding='UTF-8') as file:
                building_data = json.load(file)

            colors = {}
            polygons = []

            for feature in building_data["features"]:
                key_value = feature["properties"].get("key")
                if key_value in variables.category_color:
                    colors[key_value] = variables.category_color[key_value]

            if not building_data["features"]:
                continue
            else:
                index += 1
                gdf = gpd.read_file(building_filename)
                for key in range(len(gdf)):
                    if isinstance(gdf.loc[key, 'key'], list):
                        gdf.loc[key, 'key'] = 'hospital'
                gdf['color'] = gdf['key'].map(colors)

                xmin, ymin, xmax, ymax = (left, upper, right, lower)
                gdf_cut = gdf.cx[xmin:xmax, ymin:ymax]

                # Check if GeoDataFrame is empty
                if gdf_cut.empty:
                    print(f"{building_filename} resulted in an empty GeoDataFrame after applying filter, skipping...")
                    continue

                fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)

                with open(boundary_filename, "r", encoding='UTF-8') as file:
                    boundary_data = json.load(file)
                boundary_gdf = gpd.GeoDataFrame.from_features(boundary_data, crs="EPSG:4326")
                boundary_gdf.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3)

                gdf_cut = gdf_cut.dropna(subset=['color'])
                # Check if GeoDataFrame is empty after dropna
                if gdf_cut.empty:
                    print(f"{building_filename} resulted in an empty GeoDataFrame after dropna, skipping...")
                    continue

                gdf_cut.plot(color=gdf_cut['color'], alpha=0.5, ax=ax, edgecolor='white', linewidth=0.5)

                ax.set_axis_off()

                image_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                              f'{city_name}_dataset', 'Image',
                                              f'{city_name}_buildings_image{filesaveindex}.png')

                if index not in under10percent:
                    name_list.append(i)
                    filesaveindex += 1
                    plt.savefig(image_filename, dpi=100, transparent=True)

    # print(name_list)

    import csv

    with open(filepath.removed_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        for number in name_list:
            writer.writerow([number])

    return name_list