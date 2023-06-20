import geopandas as gpd
import pandas as pd
import json
import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath

polygons = gpd.read_file(filepath.polygon_filepath)
points = gpd.read_file(filepath.point_filepath)

joined = gpd.sjoin(points, polygons, how="left", predicate="within")

index_mapping = {}

for i, row in joined.iterrows():
    if not pd.isna(row['index_right']):
        index_mapping[i] = int(row['index_right'])

with open(filepath.polygon_filepath, 'r', encoding='UTF-8') as file:
    polygon_json = json.load(file)

with open(filepath.point_filepath, 'r', encoding='UTF-8') as file:
    point_json = json.load(file)

for source_index, target_index in index_mapping.items():
    # propertiese_to_add : point_index properties (POI)
    properties_to_add = point_json['features'][source_index]['properties']
    for key, value in properties_to_add.items():
        # 만약 해당 POI 속성이 합치고자하는 polygon data에 존재하지 않는다면 정보 추가하기
        if key not in polygon_json['features'][target_index]['properties']:
            polygon_json['features'][target_index]['properties'][key] = value

with open(filepath.combined_filepath, "w") as outfile:
    json.dump(polygon_json, outfile)

print("POI complete")

