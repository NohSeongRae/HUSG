from shapely.geometry import Polygon
import sys
import os
import json

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

# from etc.cityname import city_name

city_name = "littlerock"

def check_intersection(polygons):
    # Calculate area for each polygon
    areas = {i: polygon.area for i, polygon in enumerate(polygons)}

    # Sort polygons by area -> 오름차순
    sorted_polygons = sorted(areas.items(), key=lambda item: item[1])

    print(sorted_polygons)

    intersection_polygons = []

    for i in range(len(sorted_polygons)):
        for j in range(i + 1, len(sorted_polygons)):
            # If a smaller polygon intersects with a larger one
            if polygons[sorted_polygons[i][0]].intersects(polygons[sorted_polygons[j][0]]):
                # 작은 거 : 큰 거
                print(f"Polygon {sorted_polygons[i][0]} intersects with Polygon {sorted_polygons[j][0]}")
                intersection_polygons.append({sorted_polygons[i][0]: sorted_polygons[j][0]})

    return intersection_polygons

dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

for i in range(1, filenum + 1):
    if i == 12:
        building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                         'Buildings', f'{city_name}_buildings{i}.geojson')


        if os.path.exists(building_filename):
            with open(building_filename, "r", encoding='UTF-8') as file:
                building_data = json.load(file)

            colors = {}
            polygons = []

            for feature in building_data["features"]:
                geometry = feature["geometry"].get("coordinates")
                polygon = Polygon(geometry[0])
                polygons.append(polygon)

            intersection_polygons = check_intersection(polygons)

            print(intersection_polygons)