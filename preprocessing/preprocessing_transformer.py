import os
import geopandas as gpd
import pickle
import re
import numpy as np
from shapely.geometry import MultiLineString
import networkx as nx
from tqdm import tqdm

from gemoetry_utils import *
from general_utils import *
from building_utils import *
from plot_utils import *

# from preprocessing.gemoetry_utils import *
# from preprocessing.general_utils import *
# from preprocessing.building_utils import *
# from preprocessing.plot_utils import *

from shapely.ops import unary_union

def merge_geometries_by_index(bounding_boxs, geometries):
    # Create a dictionary to store merged geometries by unique index
    merged_geometries = {}

    # Iterate over bounding_boxs and merge geometries
    for idx, geom in bounding_boxs:
        if idx in merged_geometries:
            merged_geometries[idx].append(geom)
        else:
            merged_geometries[idx] = [geom]

    # Iterate over geometries and merge them
    for idx, geom in geometries:
        if idx in merged_geometries:
            merged_geometries[idx].append(geom)
        else:
            merged_geometries[idx] = [geom]

    # Use unary_union to merge geometries by index
    for idx, geoms in merged_geometries.items():
        try:
            merged_geometries[idx] = unary_union(geoms)
        except Exception:  # Catching the general exception here
            # If an error occurs, set bounding_boxs to just rect_polygons
            merged_geometries[idx] = [g for g in bounding_boxs if g[0] == idx][0][1]

    # Convert dictionary to list format
    result = [[idx, geom] for idx, geom in merged_geometries.items()]

    return result

unit_length = 0.04
reference_angle = 30
n_building = 30
n_unit_road = 200
n_street = 50
pad_idx = 0
building_eos_idx = n_building - 1
street_eos_idx = n_street - 1
n_street_sample = 64
n_unit_sample = 8

# city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
#               "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
#               "sanfrancisco", "miami", "seattle", "boston", "providence",
#               "neworleans", "denver", "pittsburgh", "tampa", "washington"]

# city_names = ["philadelphia", "phoenix", "portland", "richmond", "saintpaul"]
city_names = ["neworleans", "denver", "pittsburgh", "tampa", "washington"]


city_counts = {}

def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num

for city_name in city_names:
    building_index_sequences = []
    street_index_sequences = []
    unit_center_position_datasets = []
    unit_position_datasets = []
    street_unit_position_datasets = []
    unit_coords_datasets = []

    print("city_name : ", city_name)
    city_counts[city_name] = 0

    building_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_Normalized', 'Buildings')
    boundary_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_Normalized', 'Boundaries')

    # Iterate over all .geojson files in the directory
    for building_filepath in tqdm(sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key)):
        boundary_filepath = building_filepath.replace('buildings', 'boundaries')

        # Construct the full paths
        building_filename = os.path.join(building_dir_path, building_filepath)
        boundary_filename = os.path.join(boundary_dir_path, boundary_filepath)

        if os.path.exists(building_filename) and os.path.exists(boundary_filename):
            # print(building_filename)
            boundary_gdf = gpd.read_file(boundary_filename)
            building_gdf = gpd.read_file(building_filename)

            # Get building polygons for the current file and add them to the building_polygon list
            building_polygons = [row['geometry'] for idx, row in building_gdf.iterrows()]

            # if len(building_polygons) == 1:
            #     continue

            boundary_polygon = boundary_gdf.iloc[0]['geometry']

            sorted_edges = sorted_boundary_edges(boundary_polygon, unit_length)

            groups, _ = group_by_boundary_edge(building_polygons, boundary_polygon, sorted_edges)
            if not groups:
                continue

            # print("groups ", groups.keys())

            _, boundary_lines = get_boundary_building_polygon_with_index(groups, boundary_polygon, unit_length, reference_angle)

            # exception
            unique_group_indices = set([seg[0] for seg in boundary_lines])
            if len(unique_group_indices) == 1:  # one-street
                continue
            unit_roads, closest_unit_index = split_into_unit_roads(boundary_lines, unit_length)

            if len(unit_roads) >= 200:
                city_counts[city_name] += 1
                continue

            for _, segment in unit_roads:
                segment[0] = tuple(segment[0])
                segment[1] = tuple(segment[1])

            organized_data = {}
            for group_index, segment in unit_roads:
                if group_index not in organized_data:
                    organized_data[group_index] = []
                organized_data[group_index].append(segment)

            linestrings = {}
            for group_index, segments in organized_data.items():
                # Sort the segments to connect them in order
                segments.sort(key=lambda x: x[0])  # Assume that segments are in order and there are no gaps
                # Flatten the list of segments and create a linestring
                coordinates = [coord for segment in segments for coord in segment]
                linestrings[group_index] = LineString(coordinates)

            linestring_list = [[group_index, linestring] for group_index, linestring in linestrings.items()]

            nearest_linestring_for_polygon = find_nearest_linestring_for_each_polygon(building_polygons, linestring_list)
            max_distance_for_linestring = find_maximum_distance_for_each_linestring(nearest_linestring_for_polygon)

            _, box_heights, farthest_points = get_calculated_rectangle(groups, boundary_polygon, closest_unit_index, unit_length, reference_angle, linestring_list)

            # 예제 사용:
            rect_polygons = construct_rectangles(unit_roads, max_distance_for_linestring, boundary_polygon)

            geometries = create_closed_polygons(unit_roads)

            # Extract all unique indices from rect_polygons
            rect_indices = [item[0] for item in rect_polygons]

            # Filter geometries based on the indices found in rect_polygons
            filtered_geometries = [item for item in geometries if item[0] in rect_indices]

            boundary_lines = []
            for i in range(len(geometries)):
                polygon = geometries[i][1]
                boundary_lines.append([len(boundary_lines), extract_line_segments(polygon)])
            geometries = filtered_geometries

            try:
                bounding_boxs = merge_geometries_by_index(rect_polygons, geometries)
            except Exception:  # Catch any exception that occurs during the merge
                bounding_boxs = rect_polygons

            # polygons = extend_polygon(unit_roads, box_heights, farthest_points)
            # plot_polygons(polygons)

            building_polygons = get_building_polygon(building_polygons, bounding_boxs, boundary_polygon)

            building_index_sequence = []
            street_index_sequence = []
            for unit_road_idx, unit_road in enumerate(unit_roads):
                street_index_sequence.append(unit_road[0] + 1)  # unit index, street index
                is_exist = False
                for building in building_polygons:
                    if is_exist:
                        break

                    # rule 1
                    if unit_road[0] in building[1]:
                        overlaps = project_polygon_onto_linestring_full(building[2], LineString(unit_road[1]))

                        if overlaps:
                            is_exist = True

                    # rule 2
                    p1 = np.array(unit_road[1])[0]
                    p2 = np.array(unit_road[1][1])
                    v_rotated = rotated_line_90(p1, p2, unit_length)
                    # plt.plot([v_rotated[0][0], v_rotated[1][0]],
                    #          [v_rotated[0][1], v_rotated[1][1]], linewidth=1)

                    building_segments = get_segments_as_lists(building[2])

                    is_intersect = False
                    for segment in building_segments:
                        if angle_between(LineString(v_rotated), LineString(segment)) > 45:
                            if LineString(v_rotated).intersects(LineString(segment)):
                                is_exist = True

                if is_exist:
                    building_index_sequence.append(1)
                else:
                    building_index_sequence.append(0)

            building_index_sequence = np.array(building_index_sequence)

            pad_sequence = np.zeros(n_unit_road - building_index_sequence.shape[0])
            pad_sequence[:] = 2     # eos_token
            building_index_sequence = np.concatenate((building_index_sequence, pad_sequence), axis=0)

            street_index_sequence = np.array(street_index_sequence)
            pad_sequence = np.zeros((n_unit_road - street_index_sequence.shape[0]))
            if len(pad_sequence) <= 0:
                continue
            pad_sequence[0] = street_eos_idx
            pad_sequence[1:] = pad_idx
            street_index_sequence = np.concatenate((street_index_sequence, pad_sequence), axis=0)

            unit_position_dataset = np.zeros((n_unit_road, n_unit_sample, 2))
            street_position_dataset = np.zeros((n_street, n_street_sample, 2))
            street_unit_position_dataset = np.zeros((n_unit_road, n_street_sample, 2))
            unit_coords_dataset = np.zeros((n_unit_road, 2, 2))

            street_index_bool = False

            for idx, street in enumerate(boundary_lines):
                if idx + 1 >= len(street_position_dataset):
                    street_index_bool = True
                    continue
                street_position_dataset[idx + 1] = random_sample_points_on_multiple_lines(street[1], 64)

            if street_index_bool == True:
                continue

            for idx, unit_road in enumerate(unit_roads):
                p1 = np.array(unit_road[1])[0]
                p2 = np.array(unit_road[1])[1]
                unit_position_dataset[idx] = random_sample_points_on_line(p1, p2, 8)
                street_unit_position_dataset[idx] = street_position_dataset[unit_road[0] + 1]

                file_name = boundary_filename.split('\\')[-1]
                unit_coords_dataset[idx] = [[unit_road[1][0][0], unit_road[1][0][1]], [unit_road[1][1][0], unit_road[1][1][1]]]

            building_index_sequences.append(building_index_sequence)
            street_index_sequences.append(street_index_sequence)
            unit_position_datasets.append(unit_position_dataset)
            street_unit_position_datasets.append(street_unit_position_dataset)
            unit_coords_datasets.append(unit_coords_dataset)

    building_index_sequences = np.array(building_index_sequences)
    street_index_sequences = np.array(street_index_sequences)
    unit_position_datasets = np.array(unit_position_datasets)
    street_unit_position_datasets = np.array(street_unit_position_datasets)
    unit_coords_datasets = np.array(unit_coords_datasets)

    folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'train_dataset', city_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    transformer_path = os.path.join(folder_path, 'husg_transformer_dataset')

    np.savez(transformer_path,
             building_index_sequences=np.array(building_index_sequences),
             street_index_sequences=np.array(street_index_sequences),
             unit_position_datasets=np.array(unit_position_datasets),
             street_unit_position_datasets=np.array(street_unit_position_datasets),
             unit_coords_datasets=unit_coords_datasets)

    for city, count in city_counts.items():
        print(f"{city} : {count}")

    print("Total:", sum(city_counts.values()))
    print("save finish")