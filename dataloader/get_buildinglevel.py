import json
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath
from etc import variables as variables


"""
category 구분을 위한 key 값 추가
"""

building_filename = filepath.combined_filepath

with open(building_filename, "r", encoding='UTF8') as infile:
    whole_geojson_data = json.load(infile)

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


"""
1) building geojson에 category 구분을 위한 key값 추가 
2) height -> building:levels
geojson 파일에서 building의 높이를 나타내는 속성은 "building:levels"와 "height" 두가지
이 두가지를 "building:levels"라는 하나의 속성으로 합침 
ex. height: 3 -> building:levels: 3 
"""

filtered_features = []
for feature in whole_geojson_data["features"]:
    if feature["properties"].get("building:levels") != None:
        building_level = feature["properties"]["building:levels"]
        key_value = feature["properties"].get("key")
        if key_value:
            feature["properties"] = {"key": key_value}
            feature["properties"]["building:levels"] = building_level
            filtered_features.append(feature)

    if feature["properties"].get("height") != None:
        height = feature["properties"]["height"]
        key_value = feature["properties"].get("key")
        if key_value:
            feature["properties"] = {"key": key_value}
            feature["properties"]["building:levels"] = height
            filtered_features.append(feature)

whole_geojson_data["features"] = filtered_features

with open(filepath.buildinglevel_filepath, "w") as f:
    json.dump(whole_geojson_data, f)


import json

def get_buildinglevel():
    """
    각 category 별 평균 building:levels를 반환
    -> 각 category 별 평균 building:levels = 각 category 별 (building:levels가 존재하는 경우 building:levels의 총합) / (building:levels 속성이 존재하는 건물의 수)
    :return: category별 평균 높이의 list [commercial, education, emergency, financial, government, healthcare, natural, public, sport, residence]
    """

    with open(filepath.buildinglevel_filepath, 'r') as f:
        data = json.load(f)

    # category file은 각 key가 어떤 category에 할당하는지에 대해 작성해놓은 json파일
    with open(filepath.category_filepath, 'r') as f:
        category_keys = json.load(f)

    category_sums = [0] * len(category_keys)
    category_counts = [0] * len(category_keys)

    # building:levels 속성이 존재하는 경우 각 카테고리 별로 총합을 구함
    # levels : building:levels가 존재하는 경우 building:levels의 총합
    # keys : building:levels 속성이 존재하는 건물의 수
    for feature in data['features']:
        properties = feature['properties']
        if properties.get("building:levels") is not None:
            levels = properties['building:levels']
            keys = properties['key']
            if levels.isdigit():
                for i, (category, _) in enumerate(category_keys):
                    if keys in category:
                        category_sums[i] += int(levels)
                        category_counts[i] += 1
                        break

    # category 내에서 모든 building에 building:levels가 없는 경우 분모가 0이므로 예외 처리
    category_avgs = [category_sums[i] // category_counts[i] if category_counts[i] != 0 else 0 for i in range(len(category_sums))]
    return category_avgs

if __name__=='__main__':
    avg = get_buildinglevel()
    print(avg)
