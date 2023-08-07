import osmnx as ox
import json
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath

def data_download(city_name, location):
    """
    osmnx에서 road network와 city data를 다운받기
    :param city_name: city name
    :param location: exact city name
    :return: None
    """
    tags = {
        "amenity": True,
        "building": True,
        "craft": True,
        "emergency": True,
        "healthcare": True,
        "historic": True,
        "leisure": True,
        "man_made": True,
        "military": True,
        "office": True,
        "place": True,
        "shop": True,
        "sport": True,
        "tourism": True
    }

    data = {"type": "FeatureCollection", "features": []}
    # 주어진 location에 대해 osmnx에서 태그별 데이터 다운로드 후 json으로 변환
    for tag in tags:
        gdf = ox.geometries_from_place(location, {tag: tags[tag]})
        geojson_data = json.loads(gdf.to_json())

        # 주어진 geojson_data["features"]에 대해, properties중 None이 아닌 값만을 선별해서 저장
        for feature in geojson_data["features"]:
            feature["properties"] = {k: v for k, v in feature["properties"].items() if v is not None}
            data["features"].append(feature)

    # 파일 저장
    with open(filepath.data_filepath, 'w') as f:
        json.dump(data, f)

    # 유효한 tag집합에 대해 osmnx에서 road network 다운로드
    tags = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|unclassified' \
           '|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link' \
           '|living_street|service|pedestrian|road|busway"]'

    # tags = '["highway"~"unclassified|motorway|trunk|primary|secondary|tertiary|street_limited|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|residential"]'
    graph = ox.graph_from_place(location, network_type="all", custom_filter=tags)

    # 그래프를 gdf로 변환, 모든 node 제거
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

    # edge들을 linestring으로 변환
    edges["geometry"] = edges["geometry"].apply(lambda x: LineString(x))

    # geodataframe에 geometry field 추가 (맞는지 확인 부탁)
    gdf = gpd.GeoDataFrame(edges[["geometry"]], geometry="geometry")

    gdf.to_file(filepath.roads_filepath, driver="GeoJSON")

    print("Step 1: Data download completed")
