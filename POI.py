import json
import geopandas as gpd
from cityname import city_name, location
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from rtree import index

def binary_search(polygons, point):
    left, right = 0, len(polygons)
    while left < right:
        mid = (left + right) // 2
        if polygons[mid].contains(point):
            return mid
        elif point.x < polygons[mid].bounds[0]:
            right = mid
        else:
            left = mid + 1
    return None


def find_polygon_idx(point, polygons):
    for polygon_idx, polygon in enumerate(polygons):
        if point.within(polygon):
            return polygon_idx
    return None


def process_point(args):
    point, polygons, idx = args
    possible_polygons = [polygons[i] for i in idx.intersection(point.bounds)]
    possible_polygons.sort(key=lambda p: p.bounds[0])
    polygon_idx = binary_search(possible_polygons, point)
    return polygon_idx


def POI(city_name, process_point_func, num_process=5):
    ##
    point_data_filepath = city_name + '_dataset/' + city_name + "_point_data.geojson"
    polygon_data_filepath =  city_name + '_dataset/' + city_name + "_polygon_data.geojson"

    point_gdf = gpd.read_file(point_data_filepath)
    points = [row['geometry'] for _, row in point_gdf.iterrows()]

    polygon_gdf = gpd.read_file(polygon_data_filepath)
    polygons = [row['geometry'] for _, row in polygon_gdf.iterrows()]

    idx = index.Index()
    for i, polygon in enumerate(polygons):
        idx.insert(i, polygon.bounds)

    with ProcessPoolExecutor(max_workers=num_process) as executor:
        index_mapping = list(
            executor.map(process_point_func, zip(points, [polygons] * len(points), [idx] * len(points))))

    index_mapping = {idx: polygon_idx for idx, polygon_idx in enumerate(index_mapping) if polygon_idx is not None}

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

    combined_filepath =  city_name + '_dataset/' + city_name + '_polygon_data_combined.geojson'

    with open(combined_filepath, "w") as outfile:
        json.dump(polygon_json, outfile)

    print("POI 합치기 완료")


if __name__ == "__main__":
    freeze_support()
    POI(city_name, process_point, num_process=5)
