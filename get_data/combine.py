from shapely.geometry import Polygon, shape
from shapely.wkt import loads, dumps
import os
import json
import geojson
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc.cityname import city_name
from add_key import add_key

directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Combined_Buildings')

if os.path.exists(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

if not os.path.exists(directory):
    os.makedirs(directory)

add_key(city_name)

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

index = 0

for i in range(1, filenum + 1):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Buildings', f'{city_name}_buildings{i}.geojson')

    if os.path.exists(building_filename):
        index += 1
        print(building_filename)
        with open(building_filename, "r", encoding='UTF-8') as file:
            building_data = json.load(file)

        if not building_data["features"]:
            continue

        colors = {}
        polygons = []
        keys = []

        for feature in building_data['features']:
            geometry = shape(feature['geometry'])
            key = feature["properties"].get("key")

            if geometry.geom_type == 'Polygon':
                polygons.append(geometry)
                keys.append(key)

            elif geometry.geom_type == 'MultiPolygon':
                for polygon in geometry.geoms:
                    polygons.append(polygon)
                    keys.append(key)

        polygon_index_map = {dumps(p): i for i, p in enumerate(polygons)}

        polygon_strings = set(polygon_index_map.keys())

        polygons_unique = [loads(ps) for ps in polygon_strings]
        indexes_unique = [polygon_index_map[ps] for ps in polygon_strings]

        polygons = polygons_unique
        combined_polygons = polygons_unique
        combined_keys = []

        for polygon in range(len(polygons)):
            combined_keys.append(keys[indexes_unique[polygon]])

        indexes_to_remove = []
        image_to_remove = []

        overlaps_flag = False
        for k in range(len(polygons)):
            for j in range(k + 1, len(polygons)):
                if polygons[k].contains(polygons[j]):
                    indexes_to_remove.append(k)
                    if combined_keys[j] == "residence":
                        combined_keys[j] = combined_keys[k]
                elif polygons[j].contains(polygons[k]):
                    indexes_to_remove.append(j)
                    if combined_keys[k] == "residence":
                        combined_keys[k] = combined_keys[j]
                elif polygons[k].overlaps(polygons[j]):
                    overlaps_flag = True
                    break
            if overlaps_flag:
                break
        if overlaps_flag:
            pass

        indexes_to_remove = list(set(indexes_to_remove))
        indexes_to_remove.sort(reverse=True)

        for idx in indexes_to_remove:
            del combined_keys[idx]
            del combined_polygons[idx]

        geojson_polygons = [geojson.Feature(geometry=polygon, properties={"key": key}) for polygon, key in
                            zip(combined_polygons, combined_keys)]

        feature_collection = geojson.FeatureCollection(geojson_polygons)

        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                         f'{city_name}_dataset',
                                         'Combined_Buildings', f'{city_name}_buildings{i}.geojson')

        if overlaps_flag == False:
            with open(building_filename, 'w') as f:
                json.dump(feature_collection, f)
