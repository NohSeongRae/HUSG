import geopandas as gpd
import json
from concurrent.futures import ThreadPoolExecutor
from cityname import city_name

polygon_filepath = city_name + '_dataset/' + city_name + '_polygon_data.geojson'
point_filepath = city_name + '_dataset/' + city_name + '_point_data.geojson'

polygons = gpd.read_file(polygon_filepath)

points = gpd.read_file(point_filepath)

index_mapping = {}

def process_point(point):
    point_idx, point_geom = point
    for i, polygon in polygons.iterrows():
        if point_geom.within(polygon.geometry):
            return point_idx, i
    return point_idx, None

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_point, (i, point)) for i, point in points.geometry.items()]

    for future in futures:
        point_idx, polygon_idx = future.result()
        if polygon_idx is not None:
            index_mapping[point_idx] = polygon_idx

polygon_filepath = city_name + '_dataset/' + city_name + '_polygon_data.geojson'
with open(polygon_filepath, 'r', encoding='UTF-8') as file:
    polygon_json = json.load(file)
point_filepath = city_name + '_dataset/' + city_name + '_point_data.geojson'
with open(point_filepath, 'r', encoding='UTF-8') as file:
    point_json = json.load(file)
max_index = len(polygon_json['features']) - 1

for source_index, target_index in index_mapping.items():
    if target_index <= max_index:
        properties_to_add = point_json['features'][source_index]['properties']
        for key, value in properties_to_add.items():
            if key not in polygon_json['features'][target_index]['properties']:
                polygon_json['features'][target_index]['properties'][key] = value
    else:
        print(f"Skipping index {target_index} because it's out of range.")
combined_filepath = city_name + '_dataset/' + city_name + '_polygon_data_combined.geojson'
with open(combined_filepath, "w") as outfile:
    json.dump(polygon_json, outfile)
