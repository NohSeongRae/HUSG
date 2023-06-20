from shapely.geometry import Polygon
from shapely.wkt import loads, dumps
import os
import json
import geojson

city_name = "singapore"

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

index = 0

for i in range(1, filenum + 1):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Buildings', f'{city_name}_buildings{i}.geojson')
    if os.path.exists(building_filename):
        index += 1
        with open(building_filename, "r", encoding='UTF-8') as file:
            building_data = json.load(file)

        if not building_data["features"]:
            continue

        colors = {}
        polygons = []
        keys = []

        for feature in building_data["features"]:
            geometry = feature["geometry"].get("coordinates")
            key = feature["properties"].get("key")
            polygon = Polygon(geometry[0])
            polygons.append(polygon)
            keys.append(key)

        polygon_index_map = {dumps(p): i for i, p in enumerate(polygons)}

        polygon_strings = set(polygon_index_map.keys())

        polygons_unique = [loads(ps) for ps in polygon_strings]
        indexes_unique = [polygon_index_map[ps] for ps in polygon_strings]

        polygons = polygons_unique
        combined_polygons = polygons_unique
        combined_keys = []

        for i in range(len(polygons)):
            combined_keys.append(keys[indexes_unique[i]])

        indexes_to_remove = []

        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if polygons[i].contains(polygons[j]):
                    indexes_to_remove.append(i)
                    if combined_keys[j] == "residence":
                        combined_keys[j] = combined_keys[i]
                elif polygons[j].contains(polygons[i]):
                    indexes_to_remove.append(j)
                    if combined_keys[i] == "residence":
                        combined_keys[i] = combined_keys[j]

        indexes_to_remove = list(set(indexes_to_remove))
        indexes_to_remove.sort(reverse=True)

        for idx in indexes_to_remove:
            del combined_keys[idx]
            del combined_polygons[idx]

        geojson_polygons = [geojson.Feature(geometry=polygon, properties={"key": key}) for polygon, key in zip(combined_polygons, combined_keys)]

        feature_collection = geojson.FeatureCollection(geojson_polygons)

        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Combined_Buildings', f'{city_name}_buildings{index}.geojson')

        with open(building_filename, 'w') as f:
             json.dump(feature_collection, f)

import shutil

dir = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildings')
if os.path.exists(dir):
    shutil.rmtree(dir)

temp_dir = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Combined_Buildings')

os.rename(temp_dir, dir)