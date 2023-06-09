# key property 추가
import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt

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
                    if properties["building"] in ['civic']:
                        properties["key"] = "government_office"

                # shop
                if properties.get("shop") != None:
                    properties["key"] = "shop"
                    if properties["shop"] in ['convenience', 'supermarket']:
                        properties["key"] = "supermarket"
                    if properties["shop"] in ['herbalist', 'nutrition_supplements']:
                        properties["key"] = "alternative"

                # amenity
                if properties.get("amenity") != None:
                    if properties["amenity"] == 'marketplace':
                        properties["key"] = "supermarket"
                    if properties["amenity"] in ['restaurant', 'fast_food', 'cafe', 'bar', 'pub']:
                        properties["key"] = "restaurant"
                    if properties["amenity"] in ['kindergarten']:
                        properties["key"] = "kindergarten"
                    if properties["amenity"] in ['school']:
                        properties["key"] = "school"
                    if properties["amenity"] in ['college']:
                        properties["key"] = "college"
                    if properties["amenity"] in ['university']:
                        properties["key"] = "university"
                    if properties["amenity"] in ['police']:
                        properties["key"] = "police_station"
                    if properties["amenity"] in ['fire_station']:
                        properties["key"] = "fire_station"
                    if properties["amenity"] in ['bank']:
                        properties["key"] = "bank"
                    if properties["amenity"] in ['bureau_de_change']:
                        properties["key"] = "bureau_de_change"
                    if properties["amenity"] in ['court_house', 'townhall']:
                        properties["key"] = "government_office"
                    if properties["amenity"] in ['embassy']:
                        properties['key'] = 'embassy'
                    if properties["amenity"] in ['post_office']:
                        properties['key'] = 'post_office'
                    if properties["amenity"] in ['doctors']:
                        properties['key'] = 'clinic'
                    if properties["amenity"] in ['dentist']:
                        properties['key'] = 'clinic'
                    if properties["amenity"] in ['clinic']:
                        properties['key'] = 'clinic'
                    if properties["amenity"] in ['hospital']:
                        properties['key'] = 'hospital'
                    if properties["amenity"] in ['pharmacy']:
                        properties['key'] = 'pharmacy'
                    if properties["amenity"] in ['grave_yard']:
                        properties['key'] = 'cemetery'
                    if properties["amenity"] in ['place_of_worship']:
                        properties['key'] = 'place_of_worship'
                    if properties['amenity'] in ['community_centre']:
                        properties['key'] = 'community_centre'
                    if properties['amenity'] in ['library']:
                        properties['key'] = 'library'

                # office
                if properties.get("office") != None:
                    if properties["office"] in ['government']:
                        properties["key"] = 'government_office'

                # tourism
                if properties.get("tourism") != None:
                    properties["key"] = "tourism"
                    if properties["tourism"] in ['hotel', 'chalet', 'guest_house', 'hostel', 'motel']:
                        properties["key"] = "accommodation"

                # government
                if properties.get("government") != None:
                    properties["key"] = "government_office"

                # militray
                if properties.get("military") != None:
                    properties["key"] = "military"

                # landuse
                if properties.get("landuse") != None:
                    if properties["landuse"] in ['military']:
                        properties["key"] = "military"
                    if properties["landuse"] in ['cemetery']:
                        properties["key"] = "cemetery"
                    if properties["landuse"] in ['farmland', 'farmyard', 'greenhouse_horticulture']:
                        properties["key"] = "agriculture"
                    if properties["landuse"] in ['landfill']:
                        properties["key"] = "solid_waste"
                    if properties["landuse"] in ['forest']:
                        properties["key"] = "forest"
                    if properties["landuse"] in ['reservoir']:
                        properties["key"] = "reservoir"

                # health_care
                if properties.get("healthcare") != None:
                    if properties["healthcare"] in ['alternative']:
                        properties["key"] = "alternative"

                # leisure
                if properties.get("leisure") != None:
                    if properties["leisure"] in ['park']:
                        properties["key"] = "park"
                    if properties["leisure"] in ['stadium']:
                        properties["key"] = "stadium"
                    if properties["leisure"] in ['swimming_pool']:
                        properties["key"] = "swimming_pool"
                    if properties["leisure"] in ['pitch']:
                        properties["key"] = "pitch"
                    if properties["leisure"] in ['sport_centre']:
                        properties["key"] = "sport_centre"

                # natural
                if properties.get("natural") != None:
                    if properties["natural"] in ['water']:
                        properties["key"] = "water_body"
                    if properties["natural"] in ['grassland']:
                        properties["key"] = "grassland"
                    if properties["natural"] in ["wetland"]:
                        properties["key"] = "wetland"
                    if properties["natural"] in ["water"]:
                        properties["key"] = "reservoir"

                # historic
                if properties.get("historic") != None:
                    properties["key"] = "historic"

                # water
                if properties.get("water") != None:
                    if properties["water"] in ["reservoir"]:
                        properties["key"] = "reservoir"

                # waterway
                if properties.get("waterway") != None:
                    properties["key"] = "waterway"

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
                # commercial
                if key_value in ["shop", "supermarket", "restaurant", "tourism", "accommodation"]:
                    colors[key_value] = [0.9294117647058824, 0.8313725490196079, 0.5607843137254902, 1]
                # education
                elif key_value in ["kindergarten", "school", "college", "university"]:
                    colors[key_value] = [0.8901960784313725, 0.8117647058823529, 0.6549019607843137, 1]
                    # emergency
                elif key_value in ["police_station", "ambulance_station", "fire_station"]:
                    colors[key_value] = [0.9607843137254902, 0.8784313725490196, 0.8784313725490196, 1]
                # financial
                elif key_value in ["bank", "bureau_de_change"]:
                    colors[key_value] = [0.8980392156862745, 0.9019607843137255, 0.9215686274509803, 1]
                # government
                elif key_value in ["government_office", "embassy", "military", "post_office"]:
                    colors[key_value] = [1.0, 0.8509803921568627, 0.4, 1]
                # healthcare
                elif key_value in ["doctor", "dentist", "clinic", "hospital", "pharmacy", "alternative"]:
                    colors[key_value] = [0.9607843137254902, 0.8784313725490196, 0.8784313725490196, 1]
                # landuse
                elif key_value in ["park", "cemetery", "agriculture", "solid_waste"]:
                    # color = [184, 235, 173, 100]
                    colors[key_value] = [0.42745098039215684, 0.6196078431372549, 0.19607843137254902, 1]
                # natural
                elif key_value in ["forest", "grassland", "grass"]:
                    colors[key_value] = [0.42745098039215684, 0.6196078431372549, 0.19607843137254902, 1]
                # public
                elif key_value in ["place_of_worship", "community_centre", "library", "historic", "toilet"]:
                    colors[key_value] = [0.9411764705882353, 0.9019607843137255, 0.8196078431372549, 1]
                # sport
                elif key_value in ["stadium", "swimming_pool", "pitch", "sport_centre"]:
                    colors[key_value] = [0.7803921568627451, 0.7803921568627451, 0.7058823529411765, 1]
                # water
                elif key_value in ["reservoir", "waterway", "coastline", "water_body", "wetland"]:
                    colors[key_value] = [0.6, 0.8666666666666667, 1.0, 1]
                # residence
                elif key_value in ["residence"]:
                    colors[key_value] = [0.803921568627451, 0.7647058823529411, 0.7411764705882353, 1]
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