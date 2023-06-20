# key property 추가
import os
import json
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import variables as variables

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

