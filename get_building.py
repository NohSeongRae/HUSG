import json
from shapely.geometry import shape
import concurrent.futures

def process_boundary(city_name, filenum, j, data_geojson):
    boundary_filename = city_name + '_dataset/Boundaries/' + city_name + f'_boundaries{j}.geojson'

    with open(boundary_filename, "r", encoding='UTF-8') as file:
        boundary_geojson = json.load(file)

    boundary_polygon = shape(boundary_geojson["features"][0]["geometry"])
    intersections = {"type": "FeatureCollection", "features": []}

    for feature in data_geojson["features"]:
        geom = shape(feature["geometry"])

        if geom.within(boundary_polygon):
            intersections["features"].append(feature)

    building_filename = city_name + '_dataset/Buildings/' + city_name + f'_buildings{j}.geojson'
    with open(building_filename, 'w', encoding='UTF-8') as outfile:
        json.dump(intersections, outfile)

def get_building(city_name, filenum):
    data_filepath = city_name + '_dataset/' + city_name + "_polygon_data_combined.geojson"

    with open(data_filepath, "r", encoding='UTF-8') as file:
        data_geojson = json.load(file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for j in range(1, filenum):
            futures.append(executor.submit(process_boundary, city_name, filenum, j, data_geojson))

        # 결과 처리
        for future in concurrent.futures.as_completed(futures):
            result = future.result()


