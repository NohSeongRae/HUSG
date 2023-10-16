import pickle
import numpy as np
import geopandas as gpd
import re
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint, Point, MultiPolygon, MultiLineString
from scipy.spatial import Voronoi
from shapely.ops import cascaded_union
import os

unit_length = 0.04

unit_coords_path = './dataset/husg_unit_coords.pkl'

#boundary_root_path = 'Z:/iiixr-drive/Projects/2023_City_Team/dublin_dataset/Normalized/Boundaries/'
#inference_image_root_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'inference_image')


def count_files_in_directory(directory_path):
    return sum([1 for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])

def extract_number_from_string(s):
    return int(re.search(r'(\d+)(?=\.\w+$)', s).group())

def extract_numbers_from_boundaryfile(s):
    return int(re.search(r'(\d+)', s).group())

def plot(transformer_output, index):
    with open(unit_coords_path, 'rb') as f:
        unit_coords_data = pickle.load(f)

    unique_boundary_indices = list(set([seg[0] for seg in unit_coords_data]))
    sorted_boundary_indices = sorted(unique_boundary_indices, key=extract_number_from_string)
    original_boundary = sorted_boundary_indices[index]

    building_index_sequences = transformer_output

    unit_coords = []

    for unit_coord in unit_coords_data:
        if unit_coord[0] == original_boundary:
            unit_coords.append(unit_coord[1])

    building_exists_index = []

    for idx in range(len(unit_coords)):
        if building_index_sequences[idx] >= 0.5:
            building_exists_index.append(idx)

    unit_with_building = []

    for exist_idx in building_exists_index:
        unit_with_building.append(unit_coords[exist_idx])


    lines = [LineString(coords) for coords in unit_with_building]
    all_lines = [LineString(coords) for coords in unit_coords]

    coords = []
    for line in all_lines:
        coords.extend(list(line.coords[:-1]))
    whole_polygon = Polygon(coords)

    fig, ax = plt.subplots()

    x, y = whole_polygon.exterior.xy
    ax.plot(x, y, 'black')

    for line in lines:
        x, y = line.xy
        ax.plot(x, y, 'r-')

    ax.legend()
    ax.grid(True)

    fileindex = extract_numbers_from_boundaryfile(original_boundary)

    plt.savefig('./images/' + str(fileindex) + '.png')
    plt.clf()
