import os
import geopandas as gpd
import pickle
import re
import numpy as np
from shapely.geometry import MultiLineString
import networkx as nx

from gemoetry_utils import *
from general_utils import *
from building_utils import *
from plot_utils import *

# from preprocessing.gemoetry_utils import *
# from preprocessing.general_utils import *
# from preprocessing.building_utils import *
# from preprocessing.plot_utils import *

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

one_hot_building_index_set_sequences = []
building_index_set_sequences = []
building_index_sequences = []
street_index_sequences = []
building_center_position_datasets = []
unit_center_position_datasets = []
unit_position_datasets = []
street_position_datasets = []
street_unit_position_datasets = []
adj_matrices_list = []
building_polygons_datasets = []
street_multiline_datasets = []
unit_coords_datasets = []

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


city_names = ["atlanta", "barcelona", "budapest", "dallas", "dublin",
              "firenze", "houston", "lasvegas", "littlerock", "manchester",
              "milan", "minneapolis", "nottingham", "paris", "philadelphia",
              "phoenix", "portland", "richmond", "saintpaul", "sanfrancisco",
              "singapore", "toronto", "vienna", "washington", "zurich",
              "miami", "seattle", "boston", "providence", "neworleans",
              "denver", "vancouver", "pittsburgh", "tampa"]

all_building_index_sequences = []
all_building_index_set_sequences = []
all_one_hot_building_index_set_sequences = []
all_street_index_sequences = []
all_building_center_position_datasets = []
all_unit_center_position_datasets = []
all_unit_position_datasets = []
all_street_position_datasets = []
all_street_unit_position_datasets = []

def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num

for city_name in city_names:
    building_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Normalized', 'Buildings')
    boundary_dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Normalized', 'Boundaries')

    # Iterate over all .geojson files in the directory
    for building_filepath in sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key):
        boundary_filepath = building_filepath.replace('buildings', 'boundaries')

        # Construct the full paths
        building_filename = os.path.join(building_dir_path, building_filepath)
        boundary_filename = os.path.join(boundary_dir_path, boundary_filepath)

        if os.path.exists(building_filename):
            print(building_filename)
            boundary_gdf = gpd.read_file(boundary_filename)
            building_gdf = gpd.read_file(building_filename)

            # Get building polygons for the current file and add them to the building_polygon list
            building_polygons = [row['geometry'] for idx, row in building_gdf.iterrows()]
            boundary_polygon = boundary_gdf.iloc[0]['geometry']

            sorted_edges = sorted_boundary_edges(boundary_polygon, unit_length)

            groups, _ = group_by_boundary_edge(building_polygons, boundary_polygon, sorted_edges)

            # print("groups ", groups.keys())

            _, boundary_lines = get_boundary_building_polygon_with_index(groups, boundary_polygon, unit_length, reference_angle)

            # exception
            unique_group_indices = set([seg[0] for seg in boundary_lines])
            if len(unique_group_indices) == 1:  # one-street
                continue
            unit_roads, closest_unit_index = split_into_unit_roads(boundary_lines, unit_length)

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

            building_index_set_sequence = []
            building_index_sequence = []
            one_hot_building_index_set_sequence = []
            street_index_sequence = []
            for unit_road_idx, unit_road in enumerate(unit_roads):
                street_index_sequence.append(unit_road[0] + 1)  # unit index, street index
                building_set = []
                for building in building_polygons:
                    # rule 1
                    if unit_road[0] in building[1]:
                        try:
                            overlaps = project_polygon_onto_linestring_full(building[2], LineString(unit_road[1]))
                        except Exception as e:
                            continue  # Continue with the next iteration of the loop

                        if overlaps:
                            building_set.append(building[0])

                    # rule 2
                    p1 = np.array(unit_road[1])[0]
                    p2 = np.array(unit_road[1][1])
                    v_rotated = rotated_line_90(p1, p2, unit_length)
                    # plt.plot([v_rotated[0][0], v_rotated[1][0]],
                    #          [v_rotated[0][1], v_rotated[1][1]], linewidth=1)

                    building_segments = get_segments_as_lists(building[2])

                    is_intersect = False
                    for segment in building_segments:
                        if LineString(v_rotated).intersects(LineString(segment)):
                            is_intersect = True

                    if is_intersect:
                        building_set.append(building[0])
                building_set = list(set(building_set))
                if len(building_set) == 0:
                    building_index_sequence.append(0)
                else:
                    building_index_sequence.append(1)

                building_set.append(building_eos_idx)
                building_set = pad_list(building_set, n_building, 0)
                building_index_set_sequence.append(building_set)

                one_hot_building_indices = np.zeros(n_building)
                one_hot_building_indices[building_set] = 1
                one_hot_building_index_set_sequence.append(one_hot_building_indices)

            building_index_sequence = np.array(building_index_sequence)
            pad_sequence = np.zeros(n_unit_road - building_index_sequence.shape[0])
            pad_sequence[:] = 2     # eos_token
            building_index_sequence = np.concatenate((building_index_sequence, pad_sequence), axis=0)

            building_index_set_sequence = np.array(building_index_set_sequence)
            pad_sequence = np.zeros((n_unit_road - building_index_set_sequence.shape[0], n_building))
            pad_sequence[:, 0] = building_eos_idx
            building_index_set_sequence = np.concatenate((building_index_set_sequence, pad_sequence), axis=0)

            street_index_sequence = np.array(street_index_sequence)
            pad_sequence = np.zeros((n_unit_road - street_index_sequence.shape[0]))
            pad_sequence[0] = street_eos_idx
            pad_sequence[1:] = pad_idx
            street_index_sequence = np.concatenate((street_index_sequence, pad_sequence), axis=0)

            one_hot_building_index_set_sequence = np.array(one_hot_building_index_set_sequence)
            pad_sequence = np.zeros((n_unit_road - one_hot_building_index_set_sequence.shape[0], n_building))
            one_hot_building_index_set_sequence = np.concatenate((one_hot_building_index_set_sequence, pad_sequence), axis=0)

            building_center_position_dataset = np.zeros((n_building, 2))
            unit_center_position_dataset = np.zeros((n_unit_road, 2))
            unit_position_dataset = np.zeros((n_unit_road, n_unit_sample, 2))
            street_position_dataset = np.zeros((n_street, n_street_sample, 2))
            street_unit_position_dataset = np.zeros((n_unit_road, n_street_sample, 2))


            for building in building_polygons:
                building_idx = building[0]
                building_center_position_dataset[building_idx] = np.array([building[2].centroid.x, building[2].centroid.y])

            streets = []
            for street in boundary_lines:
                if len(streets) == street[0]:
                    streets.append([street[1]])
                else:
                    streets[-1] += [street[1]]

            for idx, street in enumerate(boundary_lines):
                street_position_dataset[idx + 1] = random_sample_points_on_multiple_lines(street[1], 64)
            # print(np.mean(street_position_dataset, axis=1))

            for idx, unit_road in enumerate(unit_roads):
                p1 = np.array(unit_road[1])[0]
                p2 = np.array(unit_road[1])[1]
                unit_position_dataset[idx] = random_sample_points_on_line(p1, p2, 8)
                street_unit_position_dataset[idx] = street_position_dataset[unit_road[0] + 1]

                street_idx = unit_road[0]
                for bounding_box in bounding_boxs:
                    if street_idx == bounding_box[0]:
                        v_rotated = rotated_line_90_v2(p1, p2, unit_length, scale=100)
                        intersection_points = bounding_box[1].intersection(LineString(v_rotated))
                        if not intersection_points.is_empty:
                            centroid = intersection_points.centroid
                            unit_center_position_dataset[idx] = np.array([centroid.x, centroid.y])

                file_name = boundary_filename.split('\\')[-1]
                unit_coords_datasets.append([file_name, unit_road[1]])


            building_index_sequences.append(building_index_sequence)
            building_index_set_sequences.append(building_index_set_sequence)
            one_hot_building_index_set_sequences.append(one_hot_building_index_set_sequence)
            street_index_sequences.append(street_index_sequence)
            building_center_position_datasets.append(building_center_position_dataset)
            unit_center_position_datasets.append(unit_center_position_dataset)
            unit_position_datasets.append(unit_position_dataset)
            street_position_datasets.append(street_position_dataset)
            street_unit_position_datasets.append(street_unit_position_dataset)

            adj_matrix = np.zeros((n_building + n_street, n_building + n_street))
            for unit_road_idx, unit_road in enumerate(unit_roads):
                street_index = [unit_road[0] + n_building]
                building_indices = np.unique(building_index_set_sequence[unit_road_idx-2:unit_road_idx+3])  # building edge!
                building_indices = building_indices[building_indices != pad_idx]
                building_indices = building_indices[building_indices != building_eos_idx]
                building_indices -= 1

                node_indices = np.concatenate((street_index, building_indices), axis=0)
                for node1 in node_indices:
                    for node2 in node_indices:
                        if node1 != node2:
                            adj_matrix[int(node1)][int(node2)] = 1



            street_indices = []
            for boundary_line in boundary_lines:
                if boundary_line[0] not in street_indices:
                    street_indices.append(boundary_line[0] + n_building)
            street_indices = np.unique(street_indices).tolist()
            street_indices.append(street_indices[0])

            for i in range(len(street_indices) - 1):
                adj_matrix[street_indices[i]][street_indices[i+1]] = 1
                adj_matrix[street_indices[i+1]][street_indices[i]] = 1

            adj_matrices_list.append(adj_matrix)

            building_polygons_dataset = []
            for building_polygon in building_polygons:
                building_polygons_dataset.append(building_polygon)
            building_polygons_datasets.append(building_polygons_dataset)

            street_multiline_dataset = []
            street_indices = []
            for boundary_line in boundary_lines:
                street_multiline_dataset = MultiLineString(boundary_line[1])

            street_multiline_datasets.append(street_multiline_dataset)

            def extract_number_from_string(s):
                return int(re.search(r'(\d+)(?=\.\w+$)', s).group())

            file_name = city_name + "_" + str(extract_number_from_string(boundary_filename))

            # plot_groups_with_rectangles_v7(unit_roads, bounding_boxs, building_polygons, adj_matrix, n_building, street_position_dataset, file_name)

    all_building_index_sequences.extend(building_index_sequences)
    all_building_index_set_sequences.extend(building_index_set_sequences)
    all_one_hot_building_index_set_sequences.extend(one_hot_building_index_set_sequences)
    all_street_index_sequences.extend(street_index_sequences)
    all_building_center_position_datasets.extend(building_center_position_datasets)
    all_unit_center_position_datasets.extend(unit_center_position_datasets)
    all_unit_position_datasets.extend(unit_position_datasets)
    all_street_position_datasets.extend(street_position_datasets)
    all_street_unit_position_datasets.extend(street_unit_position_datasets)

# building_index_sequences = np.array(building_index_sequences) # 0 (x), 1 (o), 2 (end)
# building_index_set_sequences = np.array(building_index_set_sequences)
# one_hot_building_index_set_sequences = np.array(one_hot_building_index_set_sequences)
# street_index_sequences = np.array(street_index_sequences)
# building_center_position_datasets = np.array(building_center_position_datasets)
# unit_center_position_datasets = np.array(unit_center_position_datasets)
# unit_position_datasets = np.array(unit_position_datasets)
# street_position_datasets = np.array(street_position_datasets)
# street_unit_position_datasets = np.array(street_unit_position_datasets)

np.savez('./dataset/husg_transformer_dataset',
         building_index_sequences=np.array(all_building_index_sequences),
         building_index_set_sequences=np.array(all_building_index_set_sequences),
         one_hot_building_index_set_sequences=np.array(all_one_hot_building_index_set_sequences),
         street_index_sequences=np.array(all_street_index_sequences),
         building_center_position_datasets=np.array(all_building_center_position_datasets),
         unit_center_position_datasets=np.array(all_unit_center_position_datasets),
         unit_position_datasets=np.array(all_unit_position_datasets),
         street_position_datasets=np.array(all_street_position_datasets),
         street_unit_position_datasets=np.array(all_street_unit_position_datasets))

graphs = [nx.from_numpy_matrix(adj_matrix) for adj_matrix in adj_matrices_list]
nx.write_gpickle(graphs, './dataset/husg_diffusion_dataset_graphs.gpickle')

with open('./dataset/husg_diffusion_building_polygons.pkl', 'wb') as file:
    pickle.dump(building_polygons_datasets, file)

with open('./dataset/husg_diffusion_street_multilines.pkl', 'wb') as file:
    pickle.dump(street_multiline_datasets, file)

with open('./dataset/husg_unit_coords.pkl', 'wb') as file:
    pickle.dump(unit_coords_datasets, file)

print('save finish!!')