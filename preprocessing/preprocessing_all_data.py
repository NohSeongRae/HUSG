import os
import geopandas as gpd
import pickle
import re
import numpy as np
from shapely.geometry import MultiLineString
import networkx as nx
from rasterio.features import geometry_mask
import rasterio
from tqdm import tqdm
import concurrent.futures

from gemoetry_utils import *
from general_utils import *
from building_utils import *
from plot_utils import *

# from preprocessing.gemoetry_utils import *
# from preprocessing.general_utils import *
# from preprocessing.building_utils import *
# from preprocessing.plot_utils import *

from shapely.ops import unary_union
from shapely.geometry import box

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
n_building = 120
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
city_names = ["atlanta"]

def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num

def process_city(city_name):
    print("Processing city:", city_name)

    adj_matrices = []
    node_features = []
    building_exist_sequences = []
    street_index_sequences = []
    unit_position_datasets = []
    street_unit_position_datasets = []
    unit_coords_datasets = []
    building_polygon_datasets = []
    building_filenames = []
    boundary_filenames = []

    building_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_rotate_normalized', 'Buildings')
    boundary_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'density20_building120_rotate_normalized', 'Boundaries')

    # Iterate over all .geojson files in the directory
    for building_filepath in tqdm(sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key)[:100]):
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
            for idx in range(len(building_polygons)):
                if len(building_polygons[idx][1]) == 0:
                    near_street_idx = get_near_street_idx(building_polygons[idx][2], boundary_lines)
                    building_polygons[idx][1].append(near_street_idx)

            cur_n_street = len(boundary_lines)
            cur_n_building = len(building_polygons)
            adj_matrix = np.eye(cur_n_street + cur_n_building)
            node_feature = np.zeros((cur_n_street + cur_n_building, 6))     # is_building, x, y, w, h, theta
            building_exist_sequence = []
            street_index_sequence = []

            for unit_road_idx, unit_road in enumerate(unit_roads):
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
                    building_exist_sequence.append(1)
                else:
                    building_exist_sequence.append(0)

            for unit_road_idx, unit_road in enumerate(unit_roads):
                street_index_sequence.append(unit_road[0] + 1)  # unit index, street index

                for building in building_polygons:
                    # rule 1
                    for street_idx in building[1]:
                        adj_matrix[street_idx][cur_n_street + building[0] - 1] = 1
                        adj_matrix[cur_n_street + building[0] - 1][street_idx] = 1

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
                                is_intersect = True

                    if is_intersect:
                        adj_matrix[unit_road[0]][cur_n_street + building[0] - 1] = 1
                        adj_matrix[cur_n_street + building[0] - 1][unit_road[0]] = 1


            for building1 in building_polygons:
                x, y, w, h, theta = get_bbox_details(building1[2].minimum_rotated_rectangle)

                node_feature[cur_n_street + building1[0] - 1] = np.array([1, x, y, w, h, theta])

                x, y = building1[2].minimum_rotated_rectangle.exterior.xy
                plt.fill(x, y)

                for building2 in building_polygons:
                    idx1 = cur_n_street + building1[0] - 1
                    idx2 = cur_n_street + building2[0] - 1

                    if adj_matrix[idx1][idx2] == 0:
                        th = 0.05
                        aabb1 = building1[2].minimum_rotated_rectangle
                        aabb2 = building2[2].minimum_rotated_rectangle

                        b1_bbox = aabb1
                        b2_bbox = aabb2

                        distance = b1_bbox.distance(b2_bbox)
                        if distance <= th:
                            if not is_building_between(building1[2], building2[2], building_polygons):
                                adj_matrix[idx1][idx2] = 1
                                adj_matrix[idx2][idx1] = 1

            street_indices = []
            for idx in range(len(boundary_lines)):
                street_index = boundary_lines[idx][0]
                node_feature[street_index] = np.array([0, 0, 0, 0, 0, 0])

                if street_index < len(boundary_lines) - 1:
                    adj_matrix[street_index][street_index+1] = 1
                    adj_matrix[street_index+1][street_index] = 1
                else:
                    adj_matrix[street_index][0] = 1
                    adj_matrix[0][street_index] = 1

            unit_position_dataset = np.zeros((len(unit_roads), n_unit_sample, 2))
            street_position_dataset = np.zeros((cur_n_street, n_street_sample, 2))
            street_unit_position_dataset = np.zeros((len(unit_roads), n_street_sample, 2))
            unit_coords_dataset = np.zeros((len(unit_roads), 2, 2))

            for idx, street in enumerate(boundary_lines):
                street_position_dataset[idx] = random_sample_points_on_multiple_lines(street[1], 64)

            for idx, unit_road in enumerate(unit_roads):
                p1 = np.array(unit_road[1])[0]
                p2 = np.array(unit_road[1])[1]
                unit_position_dataset[idx] = random_sample_points_on_line(p1, p2, 8)
                street_unit_position_dataset[idx] = street_position_dataset[unit_road[0]]

                file_name = boundary_filename.split('\\')[-1]
                unit_coords_dataset[idx] = [[unit_road[1][0][0], unit_road[1][0][1]], [unit_road[1][1][0], unit_road[1][1][1]]]

            building_polygons_coord = []
            for building in building_polygons:
                building_polygons_coord.append(np.array(building[2].exterior.coords))

            building_exist_sequences.append(building_exist_sequence)
            street_index_sequences.append(street_index_sequence)
            unit_position_datasets.append(unit_position_dataset)
            street_unit_position_datasets.append(street_unit_position_dataset)
            unit_coords_datasets.append(unit_coords_dataset)
            node_features.append(node_feature)
            adj_matrices.append(adj_matrix)
            building_polygon_datasets.append(building_polygons_coord)
            building_filenames.append(building_filepath)
            boundary_filenames.append(boundary_filepath)

            # plot_groups_with_rectangles_v7(boundary_gdf, building_gdf, unit_roads, bounding_boxs, building_polygons, adj_matrix, cur_n_street, street_position_dataset, building_filepath)

    folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', '2_transformer', 'test_dataset', city_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    datasets = [
        ('building_exist_sequences', building_exist_sequences),
        ('street_index_sequences', street_index_sequences),
        ('unit_position_datasets', unit_position_datasets),
        ('street_unit_position_datasets', street_unit_position_datasets),
        ('unit_coords_datasets', unit_coords_datasets),
        ('node_features', node_features),
        ('adj_matrices', adj_matrices),
        ('building_polygons', building_polygon_datasets),
        ('building_filenames', building_filenames),
        ('boundary_filenames', boundary_filenames)
    ]
    for name, dataset in datasets:
        filepath = os.path.join(folder_path, name + 'pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)

    print("Processing for city", city_name, "finished")

if __name__ == '__main__':
    for city_name in city_names:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process_city, city_names)