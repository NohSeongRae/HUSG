# key property 추가
import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import sys

from shapely.geometry import Polygon, shape

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import variables as variables


index_li = []
building_list = []
boundary_list = []


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


def calarearatio(city_name):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    index = 0

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

                # fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)

                with open(boundary_filename, "r", encoding='UTF-8') as file:
                    boundary_data = json.load(file)
                boundary_gdf = gpd.GeoDataFrame.from_features(boundary_data, crs="EPSG:4326")
                # boundary_gdf.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3)

                gdf_cut = gdf_cut.dropna(subset=['color'])
                # Check if GeoDataFrame is empty after dropna
                if gdf_cut.empty:
                    print(f"{building_filename} resulted in an empty GeoDataFrame after dropna, skipping...")
                    continue

                # gdf_cut.plot(color=gdf_cut['color'], alpha=0.5, ax=ax)

                # ax.set_axis_off()

                building_polygons = []

                for feature in boundary_data['features']:
                    boundary_geometry = shape(feature['geometry'])

                boundary_area = boundary_geometry.area

                for feature in building_data['features']:
                    geometry = shape(feature['geometry'])

                    if geometry.geom_type == 'Polygon':
                        building_polygons.append(geometry)

                    elif geometry.geom_type == 'MultiPolygon':
                        for polygon in geometry.geoms:
                            building_polygons.append(polygon)

                building_area = 0

                for polygon in range(len(building_polygons)):
                    building_area += building_polygons[polygon].area

                building_list.append(building_area)
                boundary_list.append(boundary_area)

                image_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                              f'{city_name}_dataset', 'Image',
                                              f'{city_name}_buildings_image{index}.png')

                index_li.append(index)

                # plt.savefig(image_filename, dpi=100, transparent=True)


def getarearatio(city_name):
    calarearatio(city_name='firenze')
    filenames = []
    for i in range(len(index_li)):
        if (building_list[i]/boundary_list[i])*100 < 10:
            filenames.append(index_li[i])

    return filenames

calarearatio(city_name='minneapolis')

count = 0

for i in range(3337):
    if (building_list[i]/boundary_list[i])*100 < 10:
        count += 1
    print(f"area ratio: {(building_list[i]/boundary_list[i])*100}, filename: {index_li[i]}")
    # print(count)
