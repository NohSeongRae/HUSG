import os
import geopandas as gpd
import pickle
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, nearest_points
from shapely import affinity
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from skimage.draw import polygon as draw_polygon

from gemoetry_utils import *
from general_utils import *
from building_utils import *
from plot_utils import *

# Load the transformed pickle file
file_path = 'transformed_block_building_info.pkl'
transformed_data = pd.read_pickle(file_path)

unit_length = 0.04
reference_angle = 30

def merge_geometries_by_index(bounding_boxs, geometries):
    merged_geometries = {}

    for idx, geom in bounding_boxs:
        if idx in merged_geometries:
            merged_geometries[idx].append(geom)
        else:
            merged_geometries[idx] = [geom]

    for idx, geom in geometries:
        if idx in merged_geometries:
            merged_geometries[idx].append(geom)
        else:
            merged_geometries[idx] = [geom]

    for idx, geoms in merged_geometries.items():
        try:
            merged_geometries[idx] = unary_union(geoms)
        except Exception:
            merged_geometries[idx] = [g for g in bounding_boxs if g[0] == idx][0][1]

    result = [[idx, geom] for idx, geom in merged_geometries.items()]
    return result


def plot_bbox(building_bboxs, unit_road_bboxs, unit_road_street_indcies):
    fig, ax = plt.subplots()
    for bbox in building_bboxs:
        x, y = bbox.exterior.xy
        ax.plot(x, y, color='red')
    for bbox in unit_road_bboxs:
        x, y = bbox.exterior.xy
        ax.plot(x, y, color='blue')
    plt.show()


def create_mask(boundary_polygon, mask_size=(224, 224)):
    # Create a blank mask
    mask = np.zeros(mask_size, dtype=np.uint8)

    # Get the coordinates of the polygon's exterior
    x, y = boundary_polygon.exterior.xy

    # Scale coordinates from [0, 1] to the size of the mask
    x = np.array(x) * (mask_size[1] - 1)
    y = np.array(y) * (mask_size[0] - 1)

    # Draw the polygon on the mask
    rr, cc = draw_polygon(y, x, mask.shape)
    mask[rr, cc] = 1

    return mask

def process_block(block_info, temp_data):
    building_polygons = [Polygon(bbox['coordinates'][0]) for bbox in block_info['buildings_bbox']]
    boundary_polygon = Polygon(block_info['block_polygon']['coordinates'][0])

    sorted_edges = sorted_boundary_edges(boundary_polygon, unit_length)
    groups, _ = group_by_boundary_edge(building_polygons, boundary_polygon, sorted_edges)

    if not groups:
        print("not groups")
        return

    _, boundary_lines = get_boundary_building_polygon_with_index(groups, boundary_polygon, unit_length, reference_angle)
    unit_roads, closest_unit_index = split_into_unit_roads(boundary_lines, unit_length)

    if len(unit_roads) >= 200:
        print("len(unit_roads) >= 200")
        return

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
        segments.sort(key=lambda x: x[0])
        coordinates = [coord for segment in segments for coord in segment]
        linestrings[group_index] = LineString(coordinates)

    linestring_list = [[group_index, linestring] for group_index, linestring in linestrings.items()]

    nearest_linestring_for_polygon = find_nearest_linestring_for_each_polygon(building_polygons, linestring_list)
    max_distance_for_linestring = find_maximum_distance_for_each_linestring(nearest_linestring_for_polygon)

    _, box_heights, farthest_points = get_calculated_rectangle(groups, boundary_polygon, closest_unit_index,
                                                               unit_length, reference_angle, linestring_list)

    rect_polygons = construct_rectangles(unit_roads, max_distance_for_linestring, boundary_polygon)
    geometries = create_closed_polygons(unit_roads)
    rect_indices = [item[0] for item in rect_polygons]
    filtered_geometries = [item for item in geometries if item[0] in rect_indices]

    boundary_lines = []
    for i in range(len(geometries)):
        polygon = geometries[i][1]
        boundary_lines.append([len(boundary_lines), extract_line_segments(polygon)])
    geometries = filtered_geometries

    try:
        bounding_boxs = merge_geometries_by_index(rect_polygons, geometries)
    except Exception:
        bounding_boxs = rect_polygons

    origin_building_polygons = building_polygons
    building_polygons = get_building_polygon(building_polygons, bounding_boxs, boundary_polygon)
    for idx in range(len(building_polygons)):
        if len(building_polygons[idx][1]) == 0:
            near_street_idx = get_near_street_idx(building_polygons[idx][2], boundary_lines)
            building_polygons[idx][1].append(near_street_idx)

    sorted_building_polygons = []
    for i in range(len(building_polygons)):
        sorted_building_polygons.append(building_polygons[i][2])

    index_map = {value: index for index, value in enumerate(sorted_building_polygons)}

    plt.figure(figsize=(6, 6))

    building_bboxs = []
    for building in building_polygons:
        building_bbox = building[2].minimum_rotated_rectangle
        building_bboxs.append(building_bbox)

    unit_road_street_indcies = []
    unit_road_bboxs = []
    for unit_road in unit_roads:
        unit_road_street_indcies.append(unit_road[0])
        unit_road_bboxs.append(expand_line_to_rectangle(unit_road[1][0], unit_road[1][1]))

    plot_bbox(building_bboxs, unit_road_bboxs, unit_road_street_indcies)

    edge_index = []

    for unit_road_idx, unit_road in enumerate(unit_roads):
        edge_index.append([unit_road_idx, unit_road_idx])
        if unit_road_idx == len(unit_roads) - 1:
            edge_index.append([unit_road_idx, 0])
            edge_index.append([0, unit_road_idx])
        else:
            edge_index.append([unit_road_idx, unit_road_idx + 1])
            edge_index.append([unit_road_idx + 1, unit_road_idx])

    scale = temp_data.get('scale_factor', 1) * 500
    print(scale)
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

                        buildnig_street_count[
                            building_bbox_idx, unit_road_street_indcies[unit_road_idx]] += 1

    for building_bbox_idx_1, building_bbox_1 in enumerate(building_bboxs):
        edge_index.append(
            [len(unit_roads) + building_bbox_idx_1, len(unit_roads) + building_bbox_idx_1])

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
                        if building_polygons[building_bbox_idx_][2].intersects(
                                LineString([center_1, center_2])):
                            is_invalid = True
                            break

                for unit_road_idx_, unit_road_ in enumerate(unit_roads):
                    if LineString(unit_road_[1]).intersects(check_line):
                        is_invalid = True
                        break

                if not is_invalid:
                    edge_index.append([len(unit_roads) + building_bbox_idx_2,
                                       len(unit_roads) + building_bbox_idx_1])
                    edge_index.append([len(unit_roads) + building_bbox_idx_1,
                                       len(unit_roads) + building_bbox_idx_2])

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
                    if min_distance > building_bbox.distance(
                            building_bboxs[cur_building_idx]):
                        min_idx = building_bbox_idx
                        min_distance = building_bbox.distance(
                            building_bboxs[cur_building_idx])

            for unit_road_bbox_idx, unit_road_bbox in enumerate(unit_road_bboxs):
                if cur_building_idx != unit_road_bbox_idx:
                    if min_distance > unit_road_bbox.distance(
                            building_bboxs[cur_building_idx]):
                        min_idx = unit_road_bbox_idx
                        min_distance = unit_road_bbox.distance(
                            building_bboxs[cur_building_idx])

            edge_index.append([count_idx, min_idx])
            edge_index.append([min_idx, count_idx])

    node_features = []
    for unit_road_bbox in unit_road_bboxs:
        unit_road_feature = get_bbox_details(unit_road_bbox)
        node_features.append(unit_road_feature)

    for building_bbox in building_bboxs:
        building_bbox_feature = get_bbox_details(building_bbox)
        node_features.append(building_bbox_feature)

    building_polygons = []
    for polygon in sorted_building_polygons:
        building_polygons.append(np.array(polygon.exterior.xy))

        x, y = polygon.exterior.xy
        plt.fill(x, y, alpha=0.8)

    plot_graph(node_features, edge_index)

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.show()

    node_features = np.array(node_features)
    edge_index = np.array(edge_index)
    unit_road_street_indcies = np.array(unit_road_street_indcies)

    # Create and save boundary mask
    boundary_mask = create_mask(boundary_polygon)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(boundary_mask, cmap='gray')
    # plt.title('Condition Image from Graph Attributes')
    # plt.axis('off')
    # plt.show()

    return {
        'node_features': node_features,
        'edge_indices': edge_index,
        'unit_road_street_indices': unit_road_street_indcies,
        'building_polygons': building_polygons,
        'boundary_mask': boundary_mask
    }

# Process each block in the transformed data
processed_blocks = []
for idx, block_info in tqdm(transformed_data.iterrows(), total=transformed_data.shape[0]):
    processed_data = process_block(block_info, transformed_data.iloc[idx])
    if processed_data:
        processed_blocks.append(processed_data)

# Save processed data
output_file = 'processed_block_building_info.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(processed_blocks, f)

print("Processing complete. Data saved to:", output_file)