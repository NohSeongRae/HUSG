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

from shapely.ops import unary_union, nearest_points
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

city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
              "philadelphia", "phoenix", "portland", "richmond", "saintpaul",
              "sanfrancisco", "miami", "seattle", "boston", "providence",
              "neworleans", "denver", "pittsburgh", "tampa", "washington"]

def sort_key(filename):
    # 파일 이름에서 숫자만 추출
    num = int(''.join(filter(str.isdigit, filename)))
    return num

def process_file(file_path):
    building_filename, boundary_filename = file_path

    if os.path.exists(building_filename) and os.path.exists(boundary_filename):
        # print(building_filename)
        boundary_gdf = gpd.read_file(boundary_filename)
        building_gdf = gpd.read_file(building_filename)

        # Get building polygons for the current file and add them to the building_polygon list
        building_polygons = [row['geometry'] for idx, row in building_gdf.iterrows()]
        building_semantics = [row['key'] for idx, row in building_gdf.iterrows()]

        # if len(building_polygons) == 1:
        #     continue

        boundary_polygon = boundary_gdf.iloc[0]['geometry']
        boundary_scale = boundary_gdf.iloc[0]['scale']

        sorted_edges = sorted_boundary_edges(boundary_polygon, unit_length)

        groups, _ = group_by_boundary_edge(building_polygons, boundary_polygon, sorted_edges)

        if not groups:
            return [False]

        # print("groups ", groups.keys())

        _, boundary_lines = get_boundary_building_polygon_with_index(groups, boundary_polygon, unit_length, reference_angle)

        # exception
        unique_group_indices = set([seg[0] for seg in boundary_lines])
        if len(unique_group_indices) == 1:  # one-street
            return [False]

        unit_roads, closest_unit_index = split_into_unit_roads(boundary_lines, unit_length)

        if len(unit_roads) >= 200:
            return [False]

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
        origin_building_polygons = building_polygons
        building_polygons = get_building_polygon(building_polygons, bounding_boxs, boundary_polygon)
        for idx in range(len(building_polygons)):
            if len(building_polygons[idx][1]) == 0:
                near_street_idx = get_near_street_idx(building_polygons[idx][2], boundary_lines)
                building_polygons[idx][1].append(near_street_idx)

        #################################################
        sorted_building_polygons = []
        for i in range(len(building_polygons)):
            sorted_building_polygons.append(building_polygons[i][2])

        # Create a dictionary that maps the elements of a to their indices in sorted_a
        index_map = {value: index for index, value in enumerate(sorted_building_polygons)}
        # Sort list b using the indices from the sorted list a
        building_semantics = sorted(building_semantics,
                                    key=lambda x: index_map[origin_building_polygons[building_semantics.index(x)]])

        #######################

        # 피규어 생성 및 크기 설정: 너비 10인치, 높이 5인치
        # plt.figure(figsize=(6, 6))

        # get building bbox
        building_bboxs = []
        for building in building_polygons:
            building_bbox = building[2].minimum_rotated_rectangle
            building_bboxs.append(building_bbox)
            x, y = building[2].exterior.xy
            plt.fill(x, y, alpha=0.8)

        # get unit road bbox and street inidces
        unit_road_street_indcies = []
        unit_road_bboxs = []
        for unit_road in unit_roads:
            unit_road_street_indcies.append(unit_road[0])
            unit_road_bboxs.append(expand_line_to_rectangle(unit_road[1][0], unit_road[1][1]))

        # plot_bbox(building_bboxs, unit_road_bboxs, unit_road_street_indcies)

        ###############

        edge_index = []

        # get unit road ring graph edge index
        for unit_road_idx, unit_road in enumerate(unit_roads):
            edge_index.append([unit_road_idx, unit_road_idx])

            if unit_road_idx == len(unit_roads) - 1:
                edge_index.append([unit_road_idx, 0])
                edge_index.append([0, unit_road_idx])
            else:
                edge_index.append([unit_road_idx, unit_road_idx + 1])
                edge_index.append([unit_road_idx + 1, unit_road_idx])

        # get unit road to building edge index
        scale = 500 * boundary_scale
        buildnig_street_count = np.zeros((len(building_bboxs), unit_road_street_indcies[-1] + 1))
        for unit_road_idx, unit_road in enumerate(unit_roads):
            unit_road_coords = unit_road[1]

            p1 = np.array(unit_road_coords[0])
            p2 = np.array(unit_road_coords[1])
            v_rotated = rotated_line_90(p1, p2, unit_length, scale=scale)

            v_rotated_start = v_rotated - np.mean((p1, p2), axis=0) + p1
            v_rotated_end = v_rotated - np.mean((p1, p2), axis=0) + p2

            for building_bbox_idx, building_bbox in enumerate(building_bboxs):
                if LineString(v_rotated).intersects(building_bbox) or \
                        LineString(v_rotated_start).intersects(building_bbox) or \
                        LineString(v_rotated_end).intersects(building_bbox):
                    check_line = LineString(nearest_points(LineString(unit_road_coords), building_bbox))

                    is_invalid = False
                    for unit_road_idx_, unit_road_ in enumerate(unit_roads):
                        if unit_road_idx + 1 < unit_road_idx_ or unit_road_idx_ < unit_road_idx - 1:
                            if LineString(unit_road_[1]).intersects(check_line):
                                is_invalid = True
                                break

                    for building_bbox_idx_, building_bbox_ in enumerate(building_bboxs):
                        if building_bbox_idx_ != building_bbox_idx:
                            if building_polygons[building_bbox_idx_][2].intersects(check_line):
                                is_invalid = True
                                break

                            center_1 = (building_bbox.centroid.x, building_bbox.centroid.y)
                            center_2 = (
                            unit_road_bboxs[unit_road_idx].centroid.x, unit_road_bboxs[unit_road_idx].centroid.y)
                            if building_polygons[building_bbox_idx_][2].intersects(LineString([center_1, center_2])):
                                is_invalid = True
                                break

                    if not is_invalid:
                        # cut edge street oriented method
                        max_street_to_building = 3
                        if buildnig_street_count[
                            building_bbox_idx, unit_road_street_indcies[unit_road_idx]] >= max_street_to_building:
                            cur_distance = LineString(unit_road_coords).distance(building_bbox)

                            for unit_road_idx_, unit_road_ in enumerate(unit_roads):
                                if [unit_road_idx_, len(unit_roads) + building_bbox_idx] in edge_index and \
                                        unit_road_street_indcies[unit_road_idx] == unit_road_street_indcies[
                                    unit_road_idx_]:
                                    distance = LineString(unit_road_[1]).distance(building_bbox)
                                    if cur_distance < distance:
                                        edge_index.remove([unit_road_idx_, len(unit_roads) + building_bbox_idx])
                                        edge_index.remove([len(unit_roads) + building_bbox_idx, unit_road_idx_])

                                        edge_index.append([unit_road_idx, len(unit_roads) + building_bbox_idx])
                                        edge_index.append([len(unit_roads) + building_bbox_idx, unit_road_idx])

                                        break
                        else:
                            edge_index.append([unit_road_idx, len(unit_roads) + building_bbox_idx])
                            edge_index.append([len(unit_roads) + building_bbox_idx, unit_road_idx])

                            buildnig_street_count[building_bbox_idx, unit_road_street_indcies[unit_road_idx]] += 1

        # get building to building edge index
        for building_bbox_idx_1, building_bbox_1 in enumerate(building_bboxs):
            edge_index.append([len(unit_roads) + building_bbox_idx_1, len(unit_roads) + building_bbox_idx_1])

            for building_bbox_idx_2, building_bbox_2 in enumerate(building_bboxs):
                if building_bbox_idx_1 >= building_bbox_idx_2:
                    continue

                check_line = LineString(nearest_points(building_bbox_1, building_bbox_2))

                is_invalid = False
                if building_bbox_1.distance(building_bbox_2) < unit_length * scale:
                    for building_bbox_idx_, building_bbox_ in enumerate(building_bboxs):
                        if building_bbox_idx_ != building_bbox_idx_1 and building_bbox_idx_ != building_bbox_idx_2:
                            if building_polygons[building_bbox_idx_][2].intersects(check_line):
                                is_invalid = True
                                break

                            center_1 = (building_bbox_1.centroid.x, building_bbox_1.centroid.y)
                            center_2 = (building_bbox_2.centroid.x, building_bbox_2.centroid.y)
                            if building_polygons[building_bbox_idx_][2].intersects(LineString([center_1, center_2])):
                                is_invalid = True
                                break

                    for unit_road_idx_, unit_road_ in enumerate(unit_roads):
                        if LineString(unit_road_[1]).intersects(check_line):
                            is_invalid = True
                            break

                    if not is_invalid:
                        edge_index.append(
                            [len(unit_roads) + building_bbox_idx_2, len(unit_roads) + building_bbox_idx_1])
                        edge_index.append(
                            [len(unit_roads) + building_bbox_idx_1, len(unit_roads) + building_bbox_idx_2])

        # get minimum building
        edge_count = np.zeros(len(unit_roads) + len(building_bboxs))
        for edge in edge_index:
            edge_count[edge[0]] += 1
            edge_count[edge[1]] += 1

        for count_idx, count in enumerate(edge_count):
            if count == 2:
                cur_building_idx = count_idx - len(unit_roads)
                min_idx = -1
                min_distance = 999
                for building_bbox_idx, building_bbox in enumerate(building_bboxs):
                    if cur_building_idx != building_bbox_idx:
                        if min_distance > building_bbox.distance(building_bboxs[cur_building_idx]):
                            min_idx = building_bbox_idx
                            min_distance = building_bbox.distance(building_bboxs[cur_building_idx])

                for unit_road_bbox_idx, unit_road_bbox in enumerate(unit_road_bboxs):
                    if cur_building_idx != unit_road_bbox_idx:
                        if min_distance > unit_road_bbox.distance(building_bboxs[cur_building_idx]):
                            min_idx = unit_road_bbox_idx
                            min_distance = unit_road_bbox.distance(building_bboxs[cur_building_idx])

                edge_index.append([count_idx, min_idx])
                edge_index.append([min_idx, count_idx])

        node_features = []
        for unit_road_bbox in unit_road_bboxs:
            unit_road_feature = get_bbox_details(unit_road_bbox)
            node_features.append(unit_road_feature)

        for building_bbox in building_bboxs:
            building_bbox_feature = get_bbox_details(building_bbox)
            node_features.append(building_bbox_feature)

        # plot_graph(node_features, edge_index)
        #
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(-0.1, 1.1)
        # plt.show()

        building_polygons = []
        for polygon in sorted_building_polygons:
            building_polygons.append(np.array(polygon.exterior.xy))

        node_features = np.array(node_features)
        edge_index = np.array(edge_index)
        unit_road_street_indcies = np.array(unit_road_street_indcies)
        building_semantics = np.array(building_semantics)

        return node_features, edge_index, unit_road_street_indcies, building_filename, \
               boundary_filename, building_polygons, building_semantics
    else:
        return [False]

if __name__ == '__main__':
    for city_name in city_names:
        print("Processing city:", city_name)

        node_features = []
        edge_indices = []
        unit_road_street_indices = []
        building_filenames = []
        boundary_filenames = []
        building_polygons = []
        building_semantics = []

        building_dir_path = os.path.join('/home', 'rhosunr99', 'HUSG', 'preprocessing', 'city_data',f'{city_name}',
                                     'density20_building120_rotate_normalized', 'Buildings')
        boundary_dir_path = os.path.join('/home', 'rhosunr99', 'HUSG', 'preprocessing', 'city_data',f'{city_name}',
                                     'density20_building120_rotate_normalized', 'Boundaries')

        input_data = []
        for building_filepath in sorted([f for f in os.listdir(building_dir_path) if f.endswith('.geojson')], key=sort_key):
            boundary_filepath = building_filepath.replace('buildings', 'boundaries')
            building_filename = os.path.join(building_dir_path, building_filepath)
            boundary_filename = os.path.join(boundary_dir_path, boundary_filepath)
            input_data.append((building_filename, boundary_filename))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = []

            # tqdm의 total 파라미터를 설정합니다.
            progress = tqdm(total=len(input_data), desc='Processing files', position=0, leave=True)

            # submit 대신 map을 사용하여 future 객체를 얻고, 각 future가 완료될 때마다 진행 상황을 업데이트합니다.
            futures = [executor.submit(process_file, data) for data in input_data]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(1)

            progress.close()

        for result in results:
            if any(x is False for x in result):
                continue

            node_feature, edge_index, unit_road_street_index, building_filename, boundary_filename, building_polygon, building_semantic = result

            node_features.append(node_feature)
            edge_indices.append(edge_index)
            unit_road_street_indices.append(unit_road_street_index)
            building_filenames.append(building_filename)
            boundary_filenames.append(boundary_filename)
            building_polygons.append(building_polygon)
            building_semantics.append(building_semantic)

        folder_path = os.path.join('/home', 'rhosunr99', 'HUSG', 'preprocessing', '2_transformer_test', 'train_dataset', city_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        datasets = [
            ('node_features', node_features),
            ('edge_indices', edge_indices),
            ('unit_road_street_indices', unit_road_street_indices),
            ('building_filenames', building_filenames),
            ('boundary_filenames', boundary_filenames),
            ('building_polygons', building_polygons),
            ('building_semantics', building_semantics)
        ]
        for name, dataset in datasets:
            filepath = os.path.join(folder_path, name + '.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f)

        print("Finished processing city:", city_name)
