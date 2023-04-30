import json
import geopandas as gpd

def POI(city_name):
    point_data_filepath = city_name + "_dataset/" + city_name + "_point_data.geojson"
    polygon_data_filepath = city_name + "_dataset/" + city_name + "_polygon_data.geojson"

    point_gdf = gpd.read_file(point_data_filepath)
    points = []
    for _, row in point_gdf.iterrows():
        points.append(row['geometry'])

    polygon_gdf = gpd.read_file(polygon_data_filepath)
    polygons = []
    for _, row in polygon_gdf.iterrows():
        polygons.append(row['geometry'])

    index_mapping = {}

    for idx, point in enumerate(points):
        found = False
        for polygon_idx, polygon in enumerate(polygons):
            if point.within(polygon):
                index_mapping[idx] = polygon_idx
                found = True
                break

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

    print("POI 합치기 완료")