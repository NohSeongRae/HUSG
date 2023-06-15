# key property 추가
import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt

import filepath
import variables

def add_key(city_name):
    # 파일 수 세기
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    # building data들 하나씩 불러오기
    for j in range(1, filenum+1):
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Buildings', f'{city_name}_buildings{j}.geojson')

        if os.path.exists(building_filename):
            with open(building_filename, "r", encoding='UTF8') as infile:
                whole_geojson_data = json.load(infile)
                print(j)
            # categorize를 위해 key값 추가하기 (key = category)
            for i in range(len(whole_geojson_data['features'])):
                properties = whole_geojson_data['features'][i]['properties']
                # building
                if properties.get("building") != None:
                    properties["key"] = "residence"
                    if properties["building"] in variables.residence:
                        properties["key"] = variables.residence[properties["building"]]
                # shop
                if properties.get("shop") != None:
                    properties["key"] = "shop"
                    if properties["shop"] in variables.shop:
                        properties["key"] = variables.shop[properties["shop"]]
                # amenity
                if properties.get("amenity") != None:
                    if properties["amenity"] in variables.amenity:
                        properties["key"] = variables.amenity[properties["amenity"]]
                # office
                if properties.get("office") != None:
                    if properties["office"] in variables.office:
                        properties["key"] = variables.office[properties["office"]]
                # tourism
                if properties.get("tourism") != None:
                    properties["key"] = "tourism"
                    if properties["tourism"] in variables.tourism:
                        properties["key"] = variables.tourism[properties["tourism"]]
                # government
                if properties.get("government") != None:
                    properties["key"] = "government_office"
                # militray
                if properties.get("military") != None:
                    properties["key"] = "military"
                # health_care
                if properties.get("healthcare") != None:
                    if properties["healthcare"] in variables.healthcare:
                        properties["key"] = variables.healthcare
                # leisure
                if properties.get("leisure") != None:
                    if properties["leisure"] in variables.leisure:
                        properties["key"] = variables.leisure[properties["leisure"]]
                # historic
                if properties.get("historic") != None:
                    properties["key"] = "historic"

            # visualize를 위해서는 key값(catgory)만 필요하므로 이에 해당하는 것들만 저장하기
            filtered_features = []
            for feature in whole_geojson_data["features"]:
                key_value = feature["properties"].get("key")
                if key_value:
                    feature["properties"] = {"key": key_value}
                    filtered_features.append(feature)

            whole_geojson_data["features"] = filtered_features

            with open(building_filename, "w") as f:
                json.dump(whole_geojson_data, f)


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
    # add_key(city_name)

    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    index = 0

    for i in range(1, filenum + 1):
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')

        boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Boundaries', f'{city_name}_boundaries{i}.geojson')

        if os.path.exists(building_filename):
            left, upper, right, lower = get_square_bounds(building_filename)

            with open(building_filename, "r", encoding='UTF-8') as file:
                building_data = json.load(file)

            # 시각화를 위한 key값(category)에 따른 색깔 할당
            colors = {}

            for feature in building_data["features"]:
                key_value = feature["properties"].get("key")
                if key_value in variables.category_color:
                    colors[key_value] = variables.category_color[key_value]
                else:
                    colors[key_value] = [1, 1, 1, 0]
                    # continue

            if not building_data["features"]:
                continue
            else:
                index += 1
                gdf = gpd.read_file(building_filename)
                # colors dictionary는 {'key':'color'}의 형태
                # key값에 따른 색상정보 저장하기
                gdf['color'] = gdf['key'].map(colors)

                xmin, ymin, xmax, ymax = (left, upper, right, lower)
                # get_square_bounds 함수에서 만든 정사각형으로 이미지 자르기
                gdf_cut = gdf.cx[xmin:xmax, ymin:ymax]
                # 원하는 크기로 시각화
                fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)

                with open(boundary_filename, "r", encoding='UTF-8') as file:
                    boundary_data = json.load(file)
                # boundary 시각화
                boundary_gdf = gpd.GeoDataFrame.from_features(boundary_data, crs="EPSG:4326")
                boundary_gdf.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3)
                # building_block 시각화
                gdf_cut.plot(color=gdf_cut['color'], alpha=0.5, ax=ax)

                ax.set_axis_off()
                # print(index, i)

                # 배경 투명으로 해서 저장

                image_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                                f'{city_name}_dataset', 'Image', f'{city_name}_buildings_image{index}.png')

                plt.savefig(image_filename, dpi=100, transparent=True)

# image('littlerock')