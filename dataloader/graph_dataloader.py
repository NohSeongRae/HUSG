import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os
import pandas as pd
import json
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath
from get_buildinglevel import get_buildinglevel
from etc.cityname import city_name

def get_building_level(semantics, json_filepath):
    levels = get_buildinglevel()

    print(levels)

    category_mapping = {
        "commercial": levels[0],
        "education": levels[1],
        "emergency": levels[2],
        "financial": levels[3],
        "government": levels[4],
        "healthcare": levels[5],
        "public": levels[6],
        "sport": levels[7],
        "residence": levels[8]
    }

    building_level_list = []

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    building_level_dict = {}
    for category_list in data:
        for item in category_list[0]:
            for category in category_list[1].split():  # assuming categories are space-separated
                if category in category_mapping:
                    building_level_dict[item] = category_mapping[category]

    building_levels = [building_level_dict.get(semantic, None) for semantic in semantics]

    # print(building_levels)

    return building_levels


def extract_polygon_coordinates(geom):
    minx, miny, maxx, maxy = geom.bounds
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2

    # use the larger range as the basis for normalization
    width = maxx - minx
    height = maxy - miny
    max_range = max(width, height)

    new_coords = []
    for point in list(geom.exterior.coords):
        lat, lon = point
        lat_norm = 2 * ((lat - midx) / max_range)
        lon_norm = 2 * ((lon - midy) / max_range)
        new_coords.append((lat_norm, lon_norm))

    return list(Polygon(new_coords).
                exterior.coords)

def graph_dataloader(city_name):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    num_file = len(files)

    centroidx_list = []
    centroidy_list = []
    group_list = []
    width_list = []
    height_list = []
    semantic_list = []
    coords_list = []
    building_level_list = []

    group = 1

    for i in range(1, num_file + 1):
        geojson_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Combined_Buildings', f'{city_name}_buildings{i}.geojson')

        if os.path.exists(geojson_filepath):
            gdf = gpd.read_file(geojson_filepath)

            polygons = []
            semantics = []
            coords = []

            for _, row in gdf.iterrows():
                geom = row['geometry']
                polygon_coords = extract_polygon_coordinates(geom)
                polygons.append(polygon_coords)
                semantic = row['key']
                semantics.append(semantic)
                coords.append(polygon_coords)

            polygon_objects = [Polygon(p) for p in polygons]
            mbrs = [p.minimum_rotated_rectangle for p in polygon_objects]

            widths = []
            heights = []

            for mbr in mbrs:
                mbr_coords = np.array(mbr.exterior.coords)
                width = np.linalg.norm(mbr_coords[0] - mbr_coords[1])
                height = np.linalg.norm(mbr_coords[1] - mbr_coords[2])
                widths.append(width)
                heights.append(height)

            centroids = [p.centroid for p in polygon_objects]

            attr_list = []

            for idx, (centroid, width, height, semantic, polygon_coords) in enumerate(
                    zip(centroids, widths, heights, semantics, coords)):
                attr = {
                    'centroid.x': centroid.x,
                    'centroid.y': centroid.y,
                    'width': width,
                    'height': height,
                    'semantic': semantic,
                    'coords': polygon_coords,
                    'group': group
                }
                attr_list.append(attr)

            centroidx_list.extend([attr['centroid.x'] for attr in attr_list])
            centroidy_list.extend([attr['centroid.y'] for attr in attr_list])
            group_list.extend([attr['group'] for attr in attr_list])
            width_list.extend([attr['width'] for attr in attr_list])
            height_list.extend([attr['height'] for attr in attr_list])
            semantic_list.extend([attr['semantic'] for attr in attr_list])
            coords_list.extend([attr['coords'] for attr in attr_list])

            building_level_list.extend(get_building_level(semantics, filepath.category_filepath))

            group += 1

    df = pd.DataFrame()
    df['centroid.x'] = centroidx_list
    df['centroid.y'] = centroidy_list
    df['group'] = group_list
    df['width'] = width_list
    df['height'] = height_list
    df['semantic'] = semantic_list
    df['coords'] = coords_list
    df['building_level'] = building_level_list

    return df

df = graph_dataloader(city_name)
df.to_csv(filepath.graph_filepath, index=False)
