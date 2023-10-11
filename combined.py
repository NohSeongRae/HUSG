import os
import numpy as np
import geopandas as gpd
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

from shapely.geometry import box

unit_length = 0.04
reference_angle = 30

from math import atan2, degrees, sqrt


def calculate_angle_from_reference(centroid, reference):
    """Calculate the angle between the centroid and a reference point."""
    dy = centroid.y - reference.y
    dx = centroid.x - reference.x
    angle = math.atan2(dy, dx)
    return angle


def distance_to_origin(point):
    """Calculate the distance between a point and the origin."""
    return point.distance(Point(0, 0))


def sort_polygons_clockwise_using_boundary_centroid_corrected(polygons, boundary_polygon):
    """Sort polygons clockwise based on the angle from the boundary polygon's centroid."""
    # Find the centroid of the boundary polygon
    boundary_polygon_centroid = boundary_polygon.centroid

    # Find the centroid closest to the origin
    closest_to_origin = min(polygons, key=lambda p: distance_to_origin(p.centroid))

    # Sort the polygons clockwise based on the angle from the boundary polygon's centroid
    sorted_polys = sorted(polygons, key=lambda p: calculate_angle_from_reference(p.centroid, boundary_polygon_centroid),
                          reverse=True)

    # Find the index of the polygon closest to the origin in the sorted list
    start_index = sorted_polys.index(closest_to_origin)

    # Rearrange the list starting from the closest polygon
    return sorted_polys[start_index:] + sorted_polys[:start_index]



def get_building_polygon(building_polygons, bounding_boxs, boundary):
    building_list = []
    building_polygons = sort_polygons_clockwise_using_boundary_centroid_corrected(building_polygons, boundary)

    # print(sorted_polygons)
    for building_idx, building_polygon in enumerate(building_polygons):
        bounding_box_indices = []
        for bounding_box in bounding_boxs:
            if bounding_box[1].intersects(building_polygon):
                bounding_box_indices.append(bounding_box[0])

        x, y = building_polygon.exterior.xy
        plt.plot(x, y, color='black')
        centroid = building_polygon.centroid
        building_text = str(building_idx+1) + ',' + ','.join(map(str, bounding_box_indices))
        plt.text(centroid.x, centroid.y, building_text, fontsize=7, ha='center', va='center', color='black')

        building_list.append([building_idx+1, bounding_box_indices, building_polygon])

    # for bounding_box in bounding_boxs:
    #     x, y = bounding_box[1].exterior.xy
    #     plt.plot(x, y, color='black')
    #     centroid = bounding_box[1].centroid
    #     text = str(bounding_box[0]+1)
    #     plt.text(centroid.x, centroid.y, text, fontsize=7, ha='center', va='center', color='black')

    return building_list

def calculate_perpendicular_distance(point, line_point1, line_point2):
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    # Calculate the line equation parameters: y = mx + b
    if x2 - x1 == 0:  # Vertical line
        # Distance from point to line is just the horizontal distance
        return abs(x0 - x1)
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Distance from point (x0, y0) to line (y = mx + b)
        return abs(m * x0 - y0 + b) / sqrt(m ** 2 + 1)


def compute_angle(p1, p2, p3):
    """Compute the angle at the middle point p2 of a triangle formed by points p1, p2, and p3."""
    # Compute vectors
    a = (p2[0] - p1[0], p2[1] - p1[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])

    # Compute dot product and magnitudes
    dot_product = a[0] * b[0] + a[1] * b[1]
    mag_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
    mag_b = math.sqrt(b[0] ** 2 + b[1] ** 2)

    # Clamp the cosine value to the range [-1, 1]
    cosine_value = dot_product / (mag_a * mag_b)
    clamped_cosine_value = max(-1.0, min(1.0, cosine_value))

    # Compute angle using dot product formula
    angle = math.acos(clamped_cosine_value)
    return math.degrees(angle)


def updated_boundary_edge_indices_v5(boundary):
    """Generate updated indices for boundary edges."""
    original_indices = list(range(len(boundary.exterior.coords) - 1))
    updated_indices = original_indices.copy()

    adjustment = 0  # 인덱스 조정값
    for i in range(1, len(original_indices)):
        b_edge = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
        if b_edge.length < unit_length:
            # 이전 선분과 동일한 인덱스를 부여
            updated_indices[i] = updated_indices[i - 1]
            adjustment += 1
        else:
            updated_indices[i] -= adjustment

    return updated_indices


def updated_boundary_edge_indices_v7(boundary):
    """Generate updated indices for boundary edges based on length and angle."""

    # Step 1: Initial index update
    updated_indices = updated_boundary_edge_indices_v5(boundary)

    # Step 2: Merge segments with the same index
    merged_segments = []
    start_idx = 0
    for i in range(1, len(updated_indices)):
        if updated_indices[i] != updated_indices[start_idx]:
            merged_segments.append((start_idx, i))
            start_idx = i
    merged_segments.append((start_idx, len(updated_indices)))

    # Step 3: Compute angle and decide on merging
    for i in range(1, len(merged_segments)):
        start1, end1 = merged_segments[i - 1]
        start2, end2 = merged_segments[i]

        # Compute angle between the two merged segments
        angle = compute_angle(boundary.exterior.coords[start1], boundary.exterior.coords[end1],
                                              boundary.exterior.coords[end2])

        if 0 <= angle <= reference_angle or 360-reference_angle <= angle <= 360:
            for j in range(start2, end2):
                updated_indices[j] = updated_indices[start1]

    # Step 4: Ensure indices are sequential
    sequential_idx = 0
    last_index = updated_indices[0]
    for i in range(len(updated_indices)):
        if updated_indices[i] != last_index:
            sequential_idx += 1
            last_index = updated_indices[i]
        updated_indices[i] = sequential_idx

    # Check for the angle between the last and the first segments
    start1, end1 = merged_segments[-1]
    start2, end2 = merged_segments[0]
    angle = compute_angle(boundary.exterior.coords[start1], boundary.exterior.coords[end1], boundary.exterior.coords[end2])
    if 0 <= angle <= 30 or 330 <= angle <= 360:
        last_index_value = updated_indices[start1]
        first_index_value = updated_indices[start2]
        for i in range(len(updated_indices)):
            if updated_indices[i] == last_index_value:
                updated_indices[i] = first_index_value

    return updated_indices


def rotated_line_90(p1, p2, scale=5):
    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    v_rotated = [(p1 + p2) / 2, (p1 + p2) / 2 + v_normalized * unit_length * scale]
    return np.array(v_rotated)


def rotated_line_90_v2(p1, p2, scale=5):
    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    v_rotated = [(p1 + p2) / 2 + v_normalized * unit_length * -scale,
                 (p1 + p2) / 2 + v_normalized * unit_length * scale]
    return np.array(v_rotated)


def random_sample_points_on_multiple_lines(lines, m):
    """
    lines: [(A1, B1), (A2, B2), ...]의 형태로 주어지는 n개의 직선
    m: 샘플링할 점의 개수
    """
    sampled_points = []

    for _ in range(m):
        # n개의 직선 중 하나를 랜덤하게 선택
        A, B = lines[np.random.randint(len(lines))]
        # 선택된 직선에서 점을 랜덤 샘플링
        point = random_sample_points_on_line(A, B, 1)[0]
        sampled_points.append(point)

    return np.array(sampled_points)


def random_sample_points_on_line(A, B, target_number):
    # 두 점 A와 B 사이의 벡터를 계산
    AB = np.array(B) - np.array(A)

    # 직선 위의 점들을 저장할 리스트 초기화
    sampled_points = []

    for _ in range(target_number):
        # 랜덤한 t 값을 [0, 1] 범위에서 선택
        t = np.random.uniform(0, 1)
        # 직선 위의 랜덤한 점을 계산
        point = A + t * AB
        sampled_points.append(point)

    return np.array(sampled_points)


def pad_list(input_list, target_length, pad_idx):
    while len(input_list) < target_length:
        input_list.append(pad_idx)
    return input_list


def get_segments_as_lists(polygon):
    # Polygon의 외곽선 좌표를 가져옴
    exterior_coords = list(polygon.exterior.coords)

    # 연속적인 좌표쌍을 사용하여 선분의 좌표 리스트 생성
    segments_coords = [[list(exterior_coords[i]), list(exterior_coords[i + 1])] for i in
                       range(len(exterior_coords) - 1)]

    return segments_coords


def updated_boundary_edge_indices_v5(boundary):
    """Generate updated indices for boundary edges."""
    original_indices = list(range(len(boundary.exterior.coords) - 1))
    updated_indices = original_indices.copy()

    adjustment = 0  # 인덱스 조정값
    for i in range(1, len(original_indices)):
        b_edge = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
        if b_edge.length < unit_length:
            # 이전 선분과 동일한 인덱스를 부여
            updated_indices[i] = updated_indices[i - 1]
            adjustment += 1
        else:
            updated_indices[i] -= adjustment

    return updated_indices


def calculate_min_bounding_box_edge_length(building_polygons):
    # Initialize a variable to store the minimum length
    min_length = float('inf')

    # Iterate through the building polygons
    for poly in building_polygons:
        # Calculate the bounding box of the polygon
        bounding_box = poly.envelope  # This gives you a rectangle (bounding box) around the polygon

        # Calculate the lengths of the bounding box edges
        x_coordinates = bounding_box.exterior.xy[0]
        y_coordinates = bounding_box.exterior.xy[1]

        # Find the minimum length among the edges of this bounding box
        min_edge_length = min(
            max(x_coordinates) - min(x_coordinates),
            max(y_coordinates) - min(y_coordinates)
        )

        # Update the minimum length if necessary
        if min_edge_length < min_length:
            min_length = min_edge_length

    return min_length


def create_unit_length_points_along_boundary(boundary, unit_length):
    """Creates points along the boundary with unit_length intervals."""

    boundary_line = boundary.exterior
    total_length = boundary_line.length
    num_points = int(total_length / unit_length)

    points = [boundary_line.interpolate(i * unit_length) for i in range(num_points)]
    return points


def create_combined_edge(boundary, edge_indices):
    combined_coords = []
    for i, idx in enumerate(edge_indices):
        if i == 0:  # For the first edge, add both start and end coordinates
            combined_coords.append(boundary.exterior.coords[idx])
        # For intermediate and last edges, just add the start coordinate
        combined_coords.append(boundary.exterior.coords[idx + 1])
    return LineString(combined_coords)


def normalize_geometry(geometry, minx, miny, maxx, maxy):
    """Normalize the coordinates of a geometry."""
    if isinstance(geometry, Polygon):
        normalized_coords = [((x - minx) / (maxx - minx), (y - miny) / (maxy - miny)) for x, y in
                             geometry.exterior.coords]
        return Polygon(normalized_coords)
    elif isinstance(geometry, LineString):
        normalized_coords = [((x - minx) / (maxx - minx), (y - miny) / (maxy - miny)) for x, y in geometry.coords]
        return LineString(normalized_coords)


from math import atan2, degrees


def create_rectangle(boundary_edge, farthest_point):
    # Extract coordinates from boundary edge
    x0, y0 = boundary_edge.coords[0]
    x1, y1 = boundary_edge.coords[-1]

    # Calculate the direction vector of the boundary_edge
    dx = x1 - x0
    dy = y1 - y0

    # Calculate the length of the boundary_edge
    length_boundary_edge = sqrt(dx ** 2 + dy ** 2)

    # Calculate unit vector of the boundary_edge
    ux = dx / length_boundary_edge
    uy = dy / length_boundary_edge

    # Calculate unit vector perpendicular to the boundary_edge (direction reversed from before)
    perp_ux = uy
    perp_uy = -ux

    # Calculate the perpendicular distance from boundary_edge to farthest_point
    distance_to_farthest_point = calculate_perpendicular_distance(
        farthest_point,
        boundary_edge.coords[0],
        boundary_edge.coords[-1]
    )

    # Calculate the coordinates of the rectangle's vertices using the unit vectors and distances
    x2 = x1 + perp_ux * distance_to_farthest_point
    y2 = y1 + perp_uy * distance_to_farthest_point

    x3 = x0 + perp_ux * distance_to_farthest_point
    y3 = y0 + perp_uy * distance_to_farthest_point

    coords = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    return Polygon(coords)


def sorted_boundary_edges(boundary):
    # Find the centroid of the boundary
    boundary_centroid = boundary.centroid

    # Create a list of boundary edges
    edges = [(i, LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])) for i in
             range(len(boundary.exterior.coords) - 1)]

    # Sort by the angle from the boundary centroid
    sorted_edges = sorted(edges, key=lambda x: calculate_angle_from_reference(Point(x[1].coords[0]), boundary_centroid))
    sorted_indices = [edge[0] for edge in sorted_edges]

    # Modify indices for edges shorter than the unit_length
    for i in range(1, len(sorted_indices)):
        b_edge = LineString(
            [boundary.exterior.coords[sorted_indices[i]], boundary.exterior.coords[sorted_indices[i] + 1]])
        if b_edge.length < unit_length:
            sorted_indices[i] = sorted_indices[i - 1]

    # Return the updated indices
    return sorted_indices


def find_polygon_with_farthest_edge(polygons, boundary_edge):
    farthest_polygon = None
    max_distance = 0

    for poly in polygons:
        for edge_start, edge_end in zip(poly.exterior.coords[:-1], poly.exterior.coords[1:]):
            edge = LineString([edge_start, edge_end])
            distance = edge.distance(boundary_edge)
            if distance > max_distance:
                max_distance = distance
                farthest_polygon = poly

    return farthest_polygon


def create_unit_length_points(line, unit_length):
    """Creates points along a line with unit_length intervals."""
    total_length = line.length
    num_points = int(total_length / unit_length)

    points = []
    for i in range(num_points):
        point = line.interpolate(i * unit_length)
        points.append(point)
    return points


def closest_boundary_edge(building, boundary, sorted_edges):
    min_distance = float('inf')
    closest_edge_index = -1

    # Get the bounds for normalization
    minx, miny, maxx, maxy = boundary.bounds

    # Normalize the building polygon
    normalized_building = normalize_geometry(building, minx, miny, maxx, maxy)

    for i in sorted_edges:
        b_edge = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])

        # Normalize the boundary edge
        normalized_b_edge = normalize_geometry(b_edge, minx, miny, maxx, maxy)

        distance = normalized_building.distance(normalized_b_edge)
        if distance < min_distance:
            min_distance = distance
            closest_edge_index = i

    return closest_edge_index


def group_by_boundary_edge(polygons, boundary, sorted_edges):
    groups = {}
    for poly in polygons:
        edge_index = closest_boundary_edge(poly, boundary, sorted_edges)
        if edge_index not in groups:
            groups[edge_index] = []
        groups[edge_index].append(poly)
    return groups


def get_unit_length_points(boundary):
    line_segment = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
    unit_length_points = create_unit_length_points(line_segment, unit_length)

    return unit_length_points


def get_continuous_groups(indices):
    groups = []
    start_idx = 0
    current_val = indices[0]

    for i, val in enumerate(indices[1:], 1):
        if val != current_val:
            groups.append((start_idx, i - 1))
            start_idx = i
            current_val = val

    groups.append((start_idx, len(indices) - 1))
    return groups

def get_boundary_building_polygon_with_index(groups, boundary):
    updated_indices = updated_boundary_edge_indices_v7(boundary)

    building_polygons = []
    boundary_lines = []
    unique_index = 0

    assigned_edges = set()
    calculated_rectangles = set()

    for original_index in groups:
        edge_index = updated_indices[original_index]
        # edge_index = original_index

        if edge_index in calculated_rectangles:
            continue

        cluster_polygons = groups[original_index]

        continuous_groups = get_continuous_groups(updated_indices)
        group_for_edge = [group for group in continuous_groups if updated_indices[group[0]] == edge_index][0]
        same_index_originals = list(range(group_for_edge[0], group_for_edge[1] + 1))

        combined_edge = create_combined_edge(boundary, same_index_originals)
        # Draw building polygon
        for poly in cluster_polygons:
            unique_index += 1
            building_polygons.append([unique_index, edge_index, poly])

        # Draw the corresponding combined boundary edge with the same color
        x, y = combined_edge.xy
        # plt.plot(x, y)

        for j in range(len(x) - 1):
            boundary_lines.append([edge_index, [[x[j], y[j]], [x[j + 1], y[j + 1]]]])

        assigned_edges.update(same_index_originals)

        calculated_rectangles.add(edge_index)

    for i in range(len(boundary.exterior.coords) - 1):
        if i not in assigned_edges:
            x, y = zip(*[boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
            boundary_index = updated_indices[i]
            for j in range(len(x) - 1):
                boundary_lines.append([boundary_index, [[x[j], y[j]], [x[j + 1], y[j + 1]]]])

    return building_polygons, boundary_lines


def get_calculated_rectangle(groups, boundary):
    updated_indices = updated_boundary_edge_indices_v7(boundary)

    calculated_rectangles = set()

    rect_polygons = []

    for original_index in groups:
        edge_index = updated_indices[original_index]

        if edge_index in calculated_rectangles:

            continue

        cluster_polygons = groups[original_index]
        farthest_polygon = find_polygon_with_farthest_edge(cluster_polygons, LineString(
            [boundary.exterior.coords[original_index], boundary.exterior.coords[original_index + 1]]))

        continuous_groups = get_continuous_groups(updated_indices)
        group_for_edge = [group for group in continuous_groups if updated_indices[group[0]] == edge_index][0]
        same_index_originals = list(range(group_for_edge[0], group_for_edge[1] + 1))

        # Create combined edge for these original indices
        combined_edge = create_combined_edge(boundary, same_index_originals)

        # Find the farthest point on the building polygon
        farthest_point = None
        max_distance = 0
        for point in farthest_polygon.exterior.coords:
            distance = combined_edge.distance(Point(point))
            if distance > max_distance:
                max_distance = distance
                farthest_point = point

        combined_edge = create_combined_edge(boundary, same_index_originals)
        rect_polygon = create_rectangle(combined_edge, farthest_point)

        rect_polygons.append([edge_index, rect_polygon])

        calculated_rectangles.add(edge_index)

    return rect_polygons


def plot_groups_with_rectangles_v7(groups, boundary):
    updated_indices = updated_boundary_edge_indices_v7(boundary)
    unique_updated_indices = list(set(updated_indices))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_updated_indices)))
    group_colors = {idx: colors[i] for i, idx in enumerate(unique_updated_indices)}

    assigned_edges = set()
    unique_index = 0
    calculated_rectangles = set()

    rect_polygons = []

    for original_index in groups:
        edge_index = updated_indices[original_index]

        if edge_index in calculated_rectangles:
            continue

        cluster_polygons = groups[original_index]
        farthest_polygon = find_polygon_with_farthest_edge(cluster_polygons, LineString(
            [boundary.exterior.coords[original_index], boundary.exterior.coords[original_index + 1]]))

        mid_point = LineString(
            [boundary.exterior.coords[original_index], boundary.exterior.coords[original_index + 1]]).centroid
        plt.text(mid_point.x, mid_point.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')
        # Get all original indices that have the same updated index
        continuous_groups = get_continuous_groups(updated_indices)
        group_for_edge = [group for group in continuous_groups if updated_indices[group[0]] == edge_index][0]
        same_index_originals = list(range(group_for_edge[0], group_for_edge[1] + 1))

        # Create combined edge for these original indices
        combined_edge = create_combined_edge(boundary, same_index_originals)

        # Find the farthest point on the building polygon
        farthest_point = None
        max_distance = 0
        for point in farthest_polygon.exterior.coords:
            distance = combined_edge.distance(Point(point))
            if distance > max_distance:
                max_distance = distance
                farthest_point = point

        combined_edge = create_combined_edge(boundary, same_index_originals)
        rect_polygon = create_rectangle(combined_edge, farthest_point)

        rect_polygons.append([edge_index, rect_polygon])

        # Draw the rectangle polygon
        rect_x, rect_y = rect_polygon.exterior.xy
        # plt.plot(rect_x, rect_y, color=group_colors[edge_index], linestyle='--', linewidth=1, alpha=0.5)

        # Draw building polygon
        # for poly in cluster_polygons:
        #     unique_index += 1
        #     x, y = poly.exterior.xy
        #     plt.plot(x, y, color=group_colors[edge_index])
        #     centroid = poly.centroid
        #     building_text = str(unique_index) + ',' + str(edge_index)
        #     # plt.text(centroid.x, centroid.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')
        #     plt.text(centroid.x, centroid.y, building_text, fontsize=7, ha='center', va='center', color='black')

        # Draw the corresponding combined boundary edge with the same color
        x, y = combined_edge.xy
        plt.plot(x, y, color=group_colors[edge_index], linewidth=1)  # thicker line for combined boundary edge
        assigned_edges.update(same_index_originals)

        calculated_rectangles.add(edge_index)

    # Draw boundary edges that are not assigned to any group
    # for i in range(len(boundary.exterior.coords) - 1):
    #     if i not in assigned_edges:
    #         x, y = zip(*[boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
    #         plt.plot(x, y, color='black', linewidth=1)
    #         mid_point = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]]).centroid
    #         plt.text(mid_point.x, mid_point.y, str(updated_indices[i]), fontsize=7, ha='center', va='center',
    #                  color='black')

    '''
    unique_index = 0
    for unit_segment in unit_segments:
        street_index, segment = unit_segment[0], unit_segment[1]
        updated_unit_index = updated_indices[street_index]
        point1, point2 = segment

        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]

        mid_pointx = (point1[0] + point2[0]) / 2
        mid_pointy = (point1[1] + point2[1]) / 2

        # group_colors에서 색상을 가져오되, street_index에 해당하는 색상이 없으면 기본값으로 검은색을 사용
        color = group_colors.get(updated_unit_index, 'black')

        plt.plot(x_values, y_values, color=color, linewidth=1)
        plt.text(mid_pointx, mid_pointy, str(unique_index), fontsize=7, ha='center', va='center', color='black')
        unique_index += 1
    '''

    # plt.show()


import math


def compute_distance(point1, point2):
    """Compute the Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def sort_boundary_lines(boundary_lines):
    # Calculate the centroid of all points
    all_coords = [coord for line in boundary_lines for coord in line[1]]
    centroid_x = sum(coord[0] for coord in all_coords) / len(all_coords)
    centroid_y = sum(coord[1] for coord in all_coords) / len(all_coords)

    # Sort by angle from centroid
    sorted_lines = sorted(boundary_lines, key=lambda x: atan2(x[1][0][1] - centroid_y, x[1][0][0] - centroid_x))

    # print(sorted_lines)

    return sorted_lines


def reverse_sort_and_connect_segments(segments):
    group0_segments = [seg for seg in segments if seg[0] == 0]
    other_segments = [seg for seg in segments if seg[0] != 0]

    # Convert segment points to tuples
    for seg in group0_segments:
        seg[1] = [tuple(point) for point in seg[1]]

    for seg in other_segments:
        seg[1] = [tuple(point) for point in seg[1]]

    start_point_dict = {seg[1][0]: seg for seg in group0_segments}
    end_point_dict = {seg[1][1]: seg for seg in group0_segments}

    start_of_group1 = other_segments[0][1][0]

    sorted_group0 = []
    current_end_point = start_of_group1

    while current_end_point in end_point_dict:
        current_seg = end_point_dict[current_end_point]
        sorted_group0.insert(0, current_seg)
        current_end_point = current_seg[1][0]

    sorted_segments = sorted_group0 + other_segments

    return sorted_segments

def distance_to_origin_from_segment(segment):
    """Calculate the minimum distance between a segment and the origin."""
    line = LineString(segment)
    point = Point(0, 0)
    return line.distance(point)

def split_into_unit_roads(boundary_lines, n):

    if boundary_lines[-1][0] == boundary_lines[0][0]:
        boundary_lines[-1][1].extend(boundary_lines[0][1])
        boundary_lines.pop(0)

    boundary_lines = reverse_sort_and_connect_segments(boundary_lines)

    # Find the group index of the segment closest to the origin
    closest_group_index = min(boundary_lines, key=lambda x: distance_to_origin_from_segment(x[1]))[0]

    # Reorder the group indices
    group_indices = sorted(set([item[0] for item in boundary_lines]))
    group_indices_reordered = group_indices[closest_group_index:] + group_indices[:closest_group_index]

    # Update the data with reordered group indices
    data_reordered = []
    for item in boundary_lines:
        new_group_index = group_indices_reordered.index(item[0])
        data_reordered.append([new_group_index, item[1]])

    boundary_lines = data_reordered

    # print(boundary_lines)

    unique_index = 0

    # print(boundary_lines)

    for line in boundary_lines:
        unique_index += 1
        point1, point2 = line[1]
        # plt.plot([point1[0], point2[0]], [point1[1], point2[1]], label=f"Line {line[0]}")

        building_text = str(line[0])
        centroid_x = (point1[0] + point2[0]) / 2
        centroid_y = (point1[1] + point2[1]) / 2

        # plt.text(centroid.x, centroid.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')
        # plt.text(centroid_x, centroid_y, building_text, fontsize=7, ha='center', va='center', color='black')

    # print(boundary_lines)

    # plt.show()
    # plt.clf()

    """Split the boundary lines into unit roads of length n."""
    unit_roads = []
    residual_length = 0
    current_line = []
    current_road_id = None

    for road_id, line in boundary_lines:
        # If it's a new road, add the current line to unit roads and reset the current line
        if current_road_id is not None and road_id != current_road_id:
            if current_line:
                unit_roads.append([current_road_id, [current_line[0], current_line[-1]]])
            current_line = []
            residual_length = 0

        current_road_id = road_id

        current_line.extend(line)

        while len(current_line) > 1:
            dist = compute_distance(current_line[0], current_line[1])
            if dist + residual_length < n:
                residual_length += dist
                current_line.pop(0)
            else:
                required_dist = n - residual_length
                ratio = required_dist / dist
                x_new = current_line[0][0] + ratio * (current_line[1][0] - current_line[0][0])
                y_new = current_line[0][1] + ratio * (current_line[1][1] - current_line[0][1])
                unit_roads.append([current_road_id, [current_line[0], [x_new, y_new]]])
                current_line[0] = [x_new, y_new]
                residual_length = 0

    if current_line:
        unit_roads.append([current_road_id, [current_line[0], current_line[-1]]])

    for i in range(len(unit_roads) - 1):
        if unit_roads[i][1][1] != unit_roads[i + 1][1][0]:
            mean = [(unit_roads[i][1][1][0] + unit_roads[i + 1][1][0][0]) / 2,
                    (unit_roads[i][1][1][1] + unit_roads[i + 1][1][0][1]) / 2]
            unit_roads[i][1][1] = unit_roads[i + 1][1][0] = mean

    # Adjusting the last point to be the mean with the first point
    if unit_roads[-1][1][1] != unit_roads[0][1][0]:
        mean = [(unit_roads[-1][1][1][0] + unit_roads[0][1][0][0]) / 2,
                (unit_roads[-1][1][1][1] + unit_roads[0][1][0][1]) / 2]
        unit_roads[-1][1][1] = unit_roads[0][1][0] = mean

    unique_index = 0
    for line in unit_roads:
        unique_index += 1
        point1, point2 = line[1]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], label=f"Line {line[0]}")
        street_index = line[0]
        building_text = str(street_index)
        # building_text = str(unique_index)
        centroid_x = (point1[0] + point2[0]) / 2
        centroid_y = (point1[1] + point2[1]) / 2

        plt.text(centroid_x, centroid_y, building_text, fontsize=7, ha='center', va='center', color='black')
        # plt.text(centroid_x, centroid_y, unique_index-1, fontsize=7, ha='center', va='center', color='black')

    # plt.show()

    return unit_roads


def project_point_onto_line(P, A, B):
    AP = P - A
    AB = B - A
    AB_squared_norm = np.sum(AB ** 2)
    scalar_proj = np.sum(AP * AB) / AB_squared_norm
    proj = A + scalar_proj * AB
    return proj


def project_polygon_onto_linestring_full(polygon, linestring):
    assert isinstance(polygon, Polygon)
    assert isinstance(linestring, LineString)

    poly_coords = np.array(polygon.exterior.coords)
    line_coords = np.array(linestring.coords)

    # Project vertices of polygon onto linestring
    proj_coords = np.array([project_point_onto_line(p, line_coords[0], line_coords[-1]) for p in poly_coords])

    # Check if the projected polygon overlaps with the linestring
    projected_polygon = Polygon(proj_coords)
    if projected_polygon.is_valid:
        overlap = projected_polygon.intersects(linestring)
    else:
        overlap = False
    return overlap


# def plot_unit_segments(unit_segments, boundary):
#     updated_indices = updated_boundary_edge_indices_v5(boundary)
#     unique_updated_indices = list(set(updated_indices))
#
#     # Calculate the total number of unique indices based on the max value in updated_indices
#     total_unique_indices = max(updated_indices) + 1
#     colors = plt.cm.rainbow(np.linspace(0, 1, total_unique_indices))
#     group_colors = {idx: colors[idx] for idx in range(total_unique_indices)}
#
#     unique_index = 0
#     for unit_segment in unit_segments:
#         street_index, segment = unit_segment[0], unit_segment[1]
#         point1, point2 = segment
#
#         x_values = [point1[0], point2[0]]
#         y_values = [point1[1], point2[1]]
#
#         mid_pointx = (point1[0] + point2[0]) / 2
#         mid_pointy = (point1[1] + point2[1]) / 2
#         color = group_colors.get(street_index, 'black')  # Use black as default if no color is assigned
#         plt.plot(x_values, y_values, color=color, linewidth=1)
#         plt.text(mid_pointx, mid_pointy, str(unique_index), fontsize=7, ha='center', va='center',
#                  color='black')
#         unique_index += 1
#
#     unique_index = 0
#
#     for original_index in groups:
#         edge_index = updated_indices[original_index]
#         cluster_polygons = groups[original_index]
#         for poly in cluster_polygons:
#             unique_index += 1
#             x, y = poly.exterior.xy
#             plt.plot(x, y, color=group_colors[edge_index])
#             centroid = poly.centroid
#             building_text = str(unique_index) + ',' + str(edge_index)
#             # plt.text(centroid.x, centroid.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')
#             plt.text(centroid.x, centroid.y, building_text, fontsize=7, ha='center', va='center', color='black')
#
#     # plt.show()


city_name = "dublin"
dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

building_polygons = []

one_hot_building_index_sequences = []
building_index_sequences = []
street_index_sequences = []
building_center_position_datasets = []
unit_center_position_datasets = []
unit_position_datasets = []
street_position_datasets = []
street_unit_position_datasets = []

for i in range(0, 200):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Normalized', 'Buildings', f'{city_name}_buildings{i}.geojson')
    boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Normalized', 'Boundaries', f'{city_name}_boundaries{i}.geojson')

    if os.path.exists(building_filename):
        print(building_filename)
        boundary_gdf = gpd.read_file(boundary_filename)
        building_gdf = gpd.read_file(building_filename)

        # Get building polygons for the current file and add them to the building_polygon list
        building_polygons = [row['geometry'] for idx, row in building_gdf.iterrows()]

        boundary_polygon = boundary_gdf.iloc[0]['geometry']

        sorted_edges = sorted_boundary_edges(boundary_polygon)
        groups = group_by_boundary_edge(building_polygons, boundary_polygon, sorted_edges)
        bounding_boxs = get_calculated_rectangle(groups, boundary_polygon)

        _, boundary_lines = get_boundary_building_polygon_with_index(groups, boundary_polygon)
        building_polygons = get_building_polygon(building_polygons, bounding_boxs, boundary_polygon)
        boundary_lines.sort(key=lambda x: x[0])

        unit_roads = split_into_unit_roads(boundary_lines, unit_length)

        n_building = 30
        n_unit_road = 200
        n_street = 50
        pad_idx = 0
        building_eos_idx = n_building - 1
        street_eos_idx = n_street - 1
        n_street_sample = 64
        n_unit_sample = 8

        building_index_sequence = []
        one_hot_building_index_sequence = []
        street_index_sequence = []
        for unit_road_idx, unit_road in enumerate(unit_roads):
            street_index_sequence.append(unit_road[0] + 1)  # unit index, street index
            building_set = []
            for building in building_polygons:
                # rule 1
                if unit_road[0] in building[1]:
                    overlaps = project_polygon_onto_linestring_full(building[2], LineString(unit_road[1]))
                    if overlaps:
                        building_set.append(building[0])

                # rule 2
                p1 = np.array(unit_road[1])[0]
                p2 = np.array(unit_road[1][1])
                v_rotated = rotated_line_90(p1, p2)
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
            building_set.append(building_eos_idx)
            building_set = pad_list(building_set, n_building, 0)
            building_index_sequence.append(building_set)

            one_hot_building_indices = np.zeros((n_building))
            one_hot_building_indices[building_set] = 1
            one_hot_building_index_sequence.append(one_hot_building_indices)

        building_index_sequence = np.array(building_index_sequence)
        pad_sequence = np.zeros((n_unit_road - building_index_sequence.shape[0], n_building))
        pad_sequence[:, 0] = building_eos_idx
        building_index_sequence = np.concatenate((building_index_sequence, pad_sequence), axis=0)

        street_index_sequence = np.array(street_index_sequence)
        pad_sequence = np.zeros((n_unit_road - street_index_sequence.shape[0]))
        pad_sequence[0] = street_eos_idx
        pad_sequence[1:] = pad_idx
        street_index_sequence = np.concatenate((street_index_sequence, pad_sequence), axis=0)

        one_hot_building_index_sequence = np.array(one_hot_building_index_sequence)
        pad_sequence = np.zeros((n_unit_road - one_hot_building_index_sequence.shape[0], n_building))
        one_hot_building_index_sequence = np.concatenate((one_hot_building_index_sequence, pad_sequence), axis=0)

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
        for idx, street in enumerate(streets):
            street_position_dataset[idx + 1] = random_sample_points_on_multiple_lines(street, 64)

        for idx, unit_road in enumerate(unit_roads):
            p1 = np.array(unit_road[1])[0]
            p2 = np.array(unit_road[1])[1]
            unit_position_dataset[idx] = random_sample_points_on_line(p1, p2, 8)
            street_unit_position_dataset[idx] = street_position_dataset[unit_road[0] + 1]

            street_idx = unit_road[0]
            for bounding_box in bounding_boxs:
                if street_idx == bounding_box[0]:
                    v_rotated = rotated_line_90_v2(p1, p2, scale=100)
                    intersection_points = bounding_box[1].intersection(LineString(v_rotated))
                    if not intersection_points.is_empty:
                        centroid = intersection_points.centroid
                        unit_center_position_dataset[idx] = np.array([centroid.x, centroid.y])

        # for i, token in enumerate(building_index_sequence):
        #     print(i, token)

        # for token in street_index_sequence:
        #     print(token)

        # for position in building_center_position_dataset:
        #     print(position)

        # for position in unit_center_position_dataset:
        #     print(position)

        # for position in unit_position_dataset:
        #     print(position)

        # for position in street_position_dataset:
        #     print(position)

        building_index_sequences.append(building_index_sequence)
        one_hot_building_index_sequences.append(one_hot_building_index_sequence)
        street_index_sequences.append(street_index_sequence)
        building_center_position_datasets.append(building_center_position_dataset)
        unit_center_position_datasets.append(unit_center_position_dataset)
        unit_position_datasets.append(unit_position_dataset)
        street_position_datasets.append(street_position_dataset)
        street_unit_position_datasets.append(street_unit_position_dataset)

        plt.show()


building_index_sequences = np.array(building_index_sequences)
one_hot_building_index_sequences = np.array(one_hot_building_index_sequences)
street_index_sequences = np.array(street_index_sequences)
building_center_position_datasets = np.array(building_center_position_datasets)
unit_center_position_datasets = np.array(unit_center_position_datasets)
unit_position_datasets = np.array(unit_position_datasets)
street_position_datasets = np.array(street_position_datasets)
street_unit_position_datasets = np.array(street_unit_position_datasets)

print(building_index_sequences.shape)  # (n, 200, 30)
print(one_hot_building_index_sequences.shape)  # (n, 200, 30)
print(street_index_sequences.shape)  # (n, 200,)
print(building_center_position_datasets.shape)  # (n, 30, 2)
print(unit_center_position_datasets.shape)  # (n, 200, 2)
print(unit_position_datasets.shape)  # (n, 200, 8, 2)
print(street_position_datasets.shape)  # (n, 50, 64, 2)
print(street_unit_position_datasets.shape)  # (n, 200, 64, 2)

np.savez('./husg_transformer_dataset',
         building_index_sequences=building_index_sequences,
         one_hot_building_index_sequences=one_hot_building_index_sequences,
         street_index_sequences=street_index_sequences,
         building_center_position_datasets=building_center_position_datasets,
         unit_center_position_datasets=unit_center_position_datasets,
         unit_position_datasets=unit_position_datasets,
         street_position_datasets=street_position_datasets,
         street_unit_position_datasets=street_unit_position_datasets)