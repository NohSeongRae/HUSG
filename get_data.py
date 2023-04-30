import os
import osmnx as ox
import json
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import shape, Point, LineString, Polygon

city_name = ["atlanta"]
location = ["Atlanta, United States"]

for i in range(len(city_name)):
    output_directories = [
        city_name[i] + "_dataset",
        city_name[i] + "_dataset/Boundaries",
        city_name[i] + "_dataset/Buildings",
    ]

    for directory in output_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 1 data download

    tags = {
        "amenity": True,
        "building": True,
        "craft": True,
        "emergency": True,
        "healthcare": True,
        "historic": True,
        "landuse": True,
        "leisure": True,
        "man_made": True,
        "military": True,
        "natural": True,
        "office": True,
        "place": True,
        "shop": True,
        "sport": True,
        "tourism": True,
        "water": True,
        "waterway": True
    }

    data = {"type": "FeatureCollection", "features": []}

    for tag in tags:
        gdf = ox.geometries_from_place(location[i], {tag: tags[tag]})
        geojson_data = json.loads(gdf.to_json())

        for feature in geojson_data["features"]:
            feature["properties"] = {k:v for k, v in feature["properties"].items() if v is not None}
            data["features"].append(feature)

    data_filepath = city_name[i] + "_dataset/" + city_name[i] + "_all_features.geojson"

    with open(data_filepath, 'w') as f:
        json.dump(data, f)

    print("data download complete")

    # point / polygon 나누기

    with open(data_filepath, "r", encoding='UTF-8') as file:
        data_geojson = json.load(file)

    point_features = {"type": "FeatureCollection", "features": []}
    polygon_features = {"type": "FeatureCollection", "features": []}

    for feature in data_geojson["features"]:
        geom_type = feature["geometry"]["type"]

        if geom_type == "Point":
            point_features["features"].append(feature)
        elif geom_type == "Polygon":
            polygon_features["features"].append(feature)

    point_data_filepath = city_name[i] + "_dataset/" + city_name[i] + "_point_data.geojson"
    with open(point_data_filepath, "w", encoding='UTF-8') as outfile:
        json.dump(point_features, outfile)

    polygon_data_filepath = city_name[i] + "_dataset/" + city_name[i] + "_polygon_data.geojson"
    with open(polygon_data_filepath, "w", encoding='UTF-8') as outfile:
        json.dump(polygon_features, outfile)

    print("extract point / polygon complete")

    # 각 point 들이 어떤 polygon에 속하는지 (POI)

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

    polygon_filepath = city_name[i] + '_dataset/' + city_name[i] + '_polygon_data.geojson'

    with open(polygon_filepath, 'r', encoding='UTF-8') as file:
        polygon_json = json.load(file)

    point_filepath = city_name[i] + '_dataset/' + city_name[i] + '_point_data.geojson'

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

    combined_filepath = city_name[i] + '_dataset/' + city_name[i] + '_polygon_data_combined.geojson'

    with open(combined_filepath, "w") as outfile:
        json.dump(polygon_json, outfile)

    print("POI 합치기 완료")

    custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"]'
    graph = ox.graph_from_place(location[i], network_type="all", simplify=False, custom_filter=custom_filter)

    simple_graph = nx.Graph(graph)
    simple_graph = simple_graph.to_undirected()

    cycles = nx.cycle_basis(simple_graph)

    polygons = []
    for cycle in cycles:
        coords = []
        for node in cycle:
            node_data = simple_graph.nodes[node]
            if 'x' in node_data and 'y' in node_data:
                coords.append((node_data['x'], node_data['y']))

        if len(coords) >= 3:
            polygon = Polygon(coords)
            polygons.append(polygon)

    for j in range(len(polygons)):
        gdf = gpd.GeoDataFrame(geometry=[polygons[i]])
        polygon_filename = city_name[i] + '_dataset/Boundaries/' + city_name[i] + f'_boundaries{i+1}.geojson'
        gdf.to_file(polygon_filename, driver='GeoJSON')

    print("Boundary 추출 완료")

    data_filepath = city_name[i] + '_dataset/' + city_name[i] + "_polygon_data_combined.geojson"

    with open(data_filepath, "r", encoding='UTF-8') as file:
        data_geojson = json.load(file)

    for j in range(1, len(polygons)):
        boundary_filename = city_name[i] + '_dataset/Boundaries/' + city_name[i] + f'_boundaries{i}.geojson'

        with open(boundary_filename, "r", encoding='UTF-8') as file:
            boundary_geojson = json.load(file)

        boundary_polygon = shape(boundary_geojson["features"][0]["geometry"])
        intersections = {"type": "FeatureCollection", "features": []}

        for feature in data_geojson["features"]:
            geom = shape(feature["geometry"])

            if geom.within(boundary_polygon):
                intersections["features"].append(feature)

        building_filename = city_name[i] + '_dataset/Buildings/' + city_name[i] + f'_buildings{i}.geojson'
        with open(building_filename, 'w', encoding='UTF-8') as outfile:
            json.dump(intersections, outfile)
