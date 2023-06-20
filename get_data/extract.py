import json
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath

def extract(city_name):
    """
    다운로드 받은 city semantic feature data들에 대해 polygon타입과 point타입을 나눈 후 따로 저장. 이는 POI.py와 get_building 진행을 위해 선행되는 작업이다.
    :param city_name: city name
    :return: None
    """

    # 기본이 되는 data인 data_geojson은 모든 feature를 담고있다.
    with open(filepath.data_filepath, "r", encoding="UTF-8") as file:
        data_geojson = json.load(file)

    # geojson data 틀
    point_features = {"type": "FeatureCollection", "features": []}
    polygon_features = {"type": "FeatureCollection", "features": []}

    for feature in data_geojson["features"]:
        geom_type = feature["geometry"]["type"]
        # geometry type이 point인 경우 amenity 정보가 있는 경우만 저장
        if geom_type == "Point":
            if feature["properties"].get("amenity"):
                point_features["features"].append(feature)
        elif geom_type == "Polygon":
            # polygon인 경우 우리가 정의한 category에 속하는 것들의 key값이 존재하는 경우만 저장
            properties = feature["properties"]
            if any(key in properties for key in
                   ["building", "shop", "amenity", "office", "tourism", "government", "military", "landuse",
                    "healthcare", "leisure", "natural", "historic", "water", "waterway"]):
                polygon_features["features"].append(feature)

    # 저장 - None data 저장을 막기 위해 json 이용
    # point feature 저장

    with open(filepath.point_filepath, "w", encoding="UTF-8") as outfile:
        json.dump(point_features, outfile)
    # polygon feature 저장

    with open(filepath.polygon_filepath, "w", encoding="UTF-8") as outfile:
        json.dump(polygon_features, outfile)

    print("Extract point / polygon data complete")
