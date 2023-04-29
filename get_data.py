# 모든 building data 다운로드

import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import networkx as nx

city_name = 'firenze'
location = "Firenze, Italy"

tags = {
    "amenity": True,
    "barrier": True,
    "building": True,
    "craft": True,
    "emergency": True,
    "geological": True,
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

dataframes = []
for tag in tags:
    gdf = ox.geometries_from_place(location, {tag: tags[tag]})
    dataframes.append(gdf)

combined_gdf = gpd.GeoDataFrame(pd.concat(dataframes, ignore_index=True))

whole_filepath = city_name + "_dataset/" + city_name + "_all_features.geojson"

combined_gdf.to_file(whole_filepath, driver="GeoJSON")

# point / polygon data features

gdf = gpd.read_file(whole_filepath)

point_filepath = city_name + "_dataset/" + city_name + "_point_data.geojson"

point_data = gdf[gdf.geometry.geom_type == "Point"]
point_data.to_file(point_filepath, driver="GeoJSON")

gdf = gpd.read_file(whole_filepath)

polygon_filepath = city_name + "_dataset/" + city_name + "_polygon_data.geojson"

polygon_data = gdf[gdf.geometry.geom_type == "Polygon"]
polygon_data.to_file(polygon_filepath, driver="GeoJSON")

import json

# null feature 제거

with open(point_filepath, "r", encoding='UTF-8') as file:
    geojson_data = json.load(file)

for feature in geojson_data["features"]:
    properties = feature["properties"]
    to_delete = [key for key, value in properties.items() if value is None]
    for key in to_delete:
        del properties[key]

point_filtered_filepath = city_name + "_dataset/" + city_name + "_point_data_filtered.geojson"

with open(point_filtered_filepath, "w") as file:
    json.dump(geojson_data, file)

with open(polygon_filepath, "r", encoding='UTF-8') as file:
    geojson_data = json.load(file)

for feature in geojson_data["features"]:
    properties = feature["properties"]
    to_delete = [key for key, value in properties.items() if value is None]
    for key in to_delete:
        del properties[key]

polygon_filtered_filepath = city_name + "_dataset/" + city_name + "_polygon_data_filtered.geojson"

with open(polygon_filtered_filepath, "w") as file:
    json.dump(geojson_data, file)

# 각 point 들이 어느 polygon에 속하는지

import json
from shapely.geometry import Point
import geopandas as gpd

point_gdf = gpd.read_file(point_filtered_filepath)

points = []

for _, row in point_gdf.iterrows():
    points.append(row['geometry'])

polygon_gdf = gpd.read_file(polygon_filtered_filepath)

polygons = []

for _, row in polygon_gdf.iterrows():
    polygons.append(row['geometry'])

# index 판단?

index_mapping = {}

for idx, point in enumerate(points):
    found = False
    for polygon_idx, polygon in enumerate(polygons):
        if point.within(polygon):
            index_mapping[idx] = polygon_idx
            found = True
            break

with open(polygon_filtered_filepath, "r", encoding='UTF-8') as file:
    polygon_json = json.load(file)

with open(point_filtered_filepath, "r", encoding='UTF-8') as file:
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

polygon_combined_filepath = city_name + "_dataset/" + city_name + "_polygon_data_combined.geojson"

with open(polygon_combined_filepath, "w") as outfile:
    json.dump(polygon_json, outfile)

# 각 polygon boundary에 해당하는 data 추출


data_filepath = city_name + "_dataset/" + city_name + "_all_features.geojson"

data = gpd.read_file(data_filepath)

# 기하학적 유형에 따라 데이터를 분리합니다.
point_data = data[data.geometry.geom_type == 'Point']
linestring_data = data[data.geometry.geom_type == 'LineString']
polygon_data = data[data.geometry.geom_type == 'Polygon']

for i in range(1, len(polygons) + 1):
    polygon_filename = city_name + "_dataset/Boundaries/" + city_name + f'_boundaries{i}.geojson'
    boundary = gpd.read_file(polygon_filename)

    # CRS 일치 여부 확인
    point_data = point_data.to_crs(boundary.crs)
    linestring_data = linestring_data.to_crs(boundary.crs)
    polygon_data = polygon_data.to_crs(boundary.crs)

    # 각 기하학적 유형에 대해 overlay를 수행하고 결과를 병합합니다.
    point_intersections = gpd.overlay(point_data, boundary, how='intersection')
    linestring_intersections = gpd.overlay(linestring_data, boundary, how='intersection')
    polygon_intersections = gpd.overlay(polygon_data, boundary, how='intersection')

    intersections = gpd.GeoDataFrame(
        pd.concat([point_intersections, linestring_intersections, polygon_intersections], ignore_index=True))

    building_filename = city_name + "_dataset/Buildings/" + city_name +  f'_building{i}.geojson'
    intersections.to_file(building_filename, driver='GeoJSON')



# road network -> cycle -> polygon

graph = ox.graph_from_place(location, network_type='all', simplify=False)

simple_graph = nx.Graph(graph)
simple_graph = simple_graph.to_undirected()

cycles = nx.cycle_basis(simple_graph)

# get cycle & polygon coordinates

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

# polygon boundary data 저장

for i in range(len(polygons)):
    gdf = gpd.GeoDataFrame(geometry=[polygons[i]])
    polygon_filename = city_name + "_dataset/Boundaries/" + city_name + f'_boundaries{i + 1}.geojson'
    gdf.to_file(polygon_filename, driver='GeoJSON')

# 각 polygon boundary에 해당하는 data 추출

data_filepath = city_name + "_dataset/" + city_name + "_all_features.geojson"

data = gpd.read_file(data_filepath)

# 기하학적 유형에 따라 데이터를 분리합니다.
point_data = data[data.geometry.geom_type == 'Point']
linestring_data = data[data.geometry.geom_type == 'LineString']
polygon_data = data[data.geometry.geom_type == 'Polygon']

for i in range(1, len(polygons) + 1):
    polygon_filename = city_name + "_dataset/Boundaries/" + city_name + f'_boundaries{i}.geojson'
    boundary = gpd.read_file(polygon_filename)

    # CRS 일치 여부 확인
    point_data = point_data.to_crs(boundary.crs)
    linestring_data = linestring_data.to_crs(boundary.crs)
    polygon_data = polygon_data.to_crs(boundary.crs)

    # 각 기하학적 유형에 대해 overlay를 수행하고 결과를 병합합니다.
    point_intersections = gpd.overlay(point_data, boundary, how='intersection')
    linestring_intersections = gpd.overlay(linestring_data, boundary, how='intersection')
    polygon_intersections = gpd.overlay(polygon_data, boundary, how='intersection')

    intersections = gpd.GeoDataFrame(
        pd.concat([point_intersections, linestring_intersections, polygon_intersections], ignore_index=True))

    building_filename = city_name + "_dataset/Buildings/" + city_name + f'_building{i}.geojson'
    intersections.to_file(building_filename, driver='GeoJSON')

for i in range(1, len(polygons) + 1):
    building_filename = city_name + "_dataset/Buildings/" + city_name + f'_building{i}.geojson'
    with open(building_filename, "r", encoding='UTF8') as infile:
        whole_geojson_data = json.load(infile)

    for i in range(len(whole_geojson_data['features'])):
        properties = whole_geojson_data['features'][i]['properties']
        # building
        if properties.get("building") != None:
            properties["key"] = "residence"
            if properties["building"] in ['civic']:
                properties["key"] = "government_office"

        # shop
        if properties.get("shop") != None:
            properties["key"] = "shop"
            if properties["shop"] in ['convenience', 'supermarket']:
                properties["key"] = "supermarket"
            if properties["shop"] in ['herbalist', 'nutrition_supplements']:
                properties["key"] = "alternative"

        # amenity
        if properties.get("amenity") != None:
            if properties["amenity"] == 'marketplace':
                properties["key"] = "supermarket"
            if properties["amenity"] in ['restaurant', 'fast_food', 'cafe', 'bar', 'pub']:
                properties["key"] = "restaurant"
            if properties["amenity"] in ['kindergarten']:
                properties["key"] = "kindergarten"
            if properties["amenity"] in ['school']:
                properties["key"] = "school"
            if properties["amenity"] in ['college']:
                properties["key"] = "college"
            if properties["amenity"] in ['university']:
                properties["key"] = "university"
            if properties["amenity"] in ['police']:
                properties["key"] = "police_station"
            if properties["amenity"] in ['fire_station']:
                properties["key"] = "fire_station"
            if properties["amenity"] in ['bank']:
                properties["key"] = "bank"
            if properties["amenity"] in ['bureau_de_change']:
                properties["key"] = "bureau_de_change"
            if properties["amenity"] in ['court_house', 'townhall']:
                properties["key"] = "government_office"
            if properties["amenity"] in ['embassy']:
                properties['key'] = 'embassy'
            if properties["amenity"] in ['post_office']:
                properties['key'] = 'post_office'
            if properties["amenity"] in ['doctors']:
                properties['key'] = 'doctor'
            if properties["amenity"] in ['dentist']:
                properties['key'] = 'dentist'
            if properties["amenity"] in ['clinic']:
                properties['key'] = 'clinic'
            if properties["amenity"] in ['hospital']:
                properties['key'] = 'hospital'
            if properties["amenity"] in ['pharmacy']:
                properties['key'] = 'pharmacy'
            if properties["amenity"] in ['grave_yard']:
                properties['key'] = 'cemetery'
            if properties["amenity"] in ['place_of_worship']:
                properties['key'] = 'place_of_worship'
            if properties['amenity'] in ['community_centre']:
                properties['key'] = 'community_centre'
            if properties['amenity'] in ['library']:
                properties['key'] = 'library'

        # office
        if properties.get("office") != None:
            if properties["office"] in ['government']:
                properties["key"] = 'government_office'

        # tourism
        if properties.get("tourism") != None:
            properties["key"] = "tourism"
            if properties["tourism"] in ['hotel', 'chalet', 'guest_house', 'hostel', 'motel']:
                properties["key"] = "accommodation"

        # government
        if properties.get("government") != None:
            properties["key"] = "government_office"

        # militray
        if properties.get("military") != None:
            properties["key"] = "military"

            # landuse
        if properties.get("landuse") != None:
            if properties["landuse"] in ['military']:
                properties["key"] = "military"
            if properties["landuse"] in ['cemetery']:
                properties["key"] = "cemetery"
            if properties["landuse"] in ['farmland', 'farmyard', 'greenhouse_horticulture']:
                properties["key"] = "agriculture"
            if properties["landuse"] in ['landfill']:
                properties["key"] = "solid_waste"
            if properties["landuse"] in ['forest']:
                properties["key"] = "forest"
            if properties["landuse"] in ['reservoir']:
                properties["key"] = "reservoir"

        # health_care
        if properties.get("healthcare") != None:
            if properties["healthcare"] in ['alternative']:
                properties["key"] = "alternative"

        # leisure
        if properties.get("leisure") != None:
            if properties["leisure"] in ['park']:
                properties["key"] = "park"
            if properties["leisure"] in ['stadium']:
                properties["key"] = "stadium"
            if properties["leisure"] in ['swimming_pool']:
                properties["key"] = "swimming_pool"
            if properties["leisure"] in ['pitch']:
                properties["key"] = "pitch"
            if properties["leisure"] in ['sport_centre']:
                properties["key"] = "sport_centre"

        # natural
        if properties.get("natural") != None:
            if properties["natural"] in ['water']:
                properties["key"] = "water_body"
            if properties["natural"] in ['grassland']:
                properties["key"] = "grassland"
            if properties["natural"] in ["wetland"]:
                properties["key"] = "wetland"
            if properties["natural"] in ["water"]:
                properties["key"] = "reservoir"

        # historic
        if properties.get("historic") != None:
            properties["key"] = "historic"

        # water
        if properties.get("water") != None:
            if properties["water"] in ["reservoir"]:
                properties["key"] = "reservoir"

        # waterway
        if properties.get("waterway") != None:
            properties["key"] = "waterway"

    with open(building_filename, "w") as f:
        json.dump(whole_geojson_data, f)