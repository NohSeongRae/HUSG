from shapely.geometry import Polygon
from shapely.wkt import loads, dumps
import os
import json

city_name = "littlerock"

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

for i in range(1, filenum + 1):
    if i == 33:
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')


        if os.path.exists(building_filename):
            with open(building_filename, "r", encoding='UTF-8') as file:
                building_data = json.load(file)

            colors = {}
            polygons = []
            keys = []

            for feature in building_data["features"]:
                geometry = feature["geometry"].get("coordinates")
                key = feature["properties"].get("key")
                polygon = Polygon(geometry[0])
                polygons.append(polygon)
                keys.append(key)

# 각 폴리곤 문자열에 대해 원래의 인덱스를 기록하는 사전을 생성합니다.
polygon_index_map = {dumps(p): i for i, p in enumerate(polygons)}

# 중복된 폴리곤을 제거합니다.
polygon_strings = set(polygon_index_map.keys())

# 중복을 제거한 폴리곤과 그에 대응하는 원래의 인덱스를 저장하는 리스트를 생성합니다.
polygons_unique = [loads(ps) for ps in polygon_strings]
indexes_unique = [polygon_index_map[ps] for ps in polygon_strings]

print(indexes_unique)

polygons = polygons_unique

for i in range(len(polygons)):
    for j in range(i + 1, len(polygons)):
        if polygons[i].contains(polygons[j]):
            print(f"Polygon {indexes_unique[i+1]} contains Polygon {indexes_unique[j+1]}")

            # print(f"Polygon {i + 1} contains Polygon {j + 1}")

        elif polygons[j].contains(polygons[i]):
            print(f"Polygon {indexes_unique[j+1]} contains Polygon {indexes_unique[i + 1]}")

            # print(f"Polygon {j + 1} contains Polygon {i + 1}")
