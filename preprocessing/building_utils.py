import math
from math import atan2
from shapely.geometry import LineString, Point

from preprocessing.gemoetry_utils import distance_to_origin, calculate_angle_from_reference, compute_angle, \
    find_polygon_with_farthest_edge, normalize_geometry, create_combined_edge, create_rectangle
from preprocessing.general_utils import compute_distance

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

    for building_idx, building_polygon in enumerate(building_polygons):
        bounding_box_indices = []
        for bounding_box in bounding_boxs:
            if bounding_box[1].intersects(building_polygon):
                bounding_box_indices.append(bounding_box[0])

        building_list.append([building_idx+1, bounding_box_indices, building_polygon])

    return building_list

def updated_boundary_edge_indices_v5(boundary, unit_length):
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

def updated_boundary_edge_indices_v7(boundary, unit_length, reference_angle):
    """Generate updated indices for boundary edges based on length and angle."""

    # Step 1: Initial index update
    updated_indices = updated_boundary_edge_indices_v5(boundary, unit_length)

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

def sorted_boundary_edges(boundary, unit_length):
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
    # Calculate centroid of the boundary
    centroid = boundary.centroid

    # Extracting boundary coordinates from the Polygon object
    boundary_coords = list(boundary.boundary.coords[:-1])  # exclude the repeated last point

    # Calculate distance and angle for each boundary point from centroid
    distances_and_angles = []
    for coord in boundary_coords:
        dx, dy = coord[0] - centroid.x, coord[1] - centroid.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx)
        distances_and_angles.append((distance, angle, coord))

    # Sort by distance, then by angle (for clock-wise arrangement)
    distances_and_angles.sort(key=lambda x: (x[0], -x[1]))

    # Sorted boundary coords based on distance and angle
    sorted_boundary_coords = [item[2] for item in distances_and_angles]

    # Create boundary info with sorted coords
    boundary_info = {i: (sorted_boundary_coords[i], sorted_boundary_coords[(i + 1) % len(sorted_boundary_coords)])
                     for i in range(len(sorted_boundary_coords))}

    groups = {}
    for poly in polygons:
        edge_index = closest_boundary_edge(poly, boundary, list(boundary_info.keys()))
        if edge_index not in groups:
            groups[edge_index] = []
        groups[edge_index].append(poly)

    return groups, boundary_info

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

def get_boundary_building_polygon_with_index(groups, boundary, unit_length, reference_angle):
    updated_indices = updated_boundary_edge_indices_v7(boundary, unit_length, reference_angle)

    building_polygons = []
    boundary_lines = []
    unique_index = 0

    assigned_edges = set()
    calculated_rectangles = set()

    for original_index in groups:
        edge_index = updated_indices[original_index]

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

    boundary_lines.sort(key=lambda x: x[0])
    return building_polygons, boundary_lines

def get_calculated_rectangle(groups, boundary, closest_index, unit_length, reference_angle):
    updated_indices = updated_boundary_edge_indices_v7(boundary, unit_length, reference_angle)

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

    rect_index = []
    for i in range(len(rect_polygons)):
        rect_index.append(rect_polygons[i][0])

    if closest_index != 0:
        for i in range(len(rect_polygons)):
            if rect_polygons[i][0] == closest_index:
                rect_polygons[i][0] = 0
            elif closest_index < rect_polygons[i][0] <= max(rect_index):
                rect_polygons[i][0] -= closest_index
            else:
                rect_polygons[i][0] += (max(rect_index) - closest_index + 1)

    return rect_polygons

def sort_boundary_lines(boundary_lines):
    # Calculate the centroid of all points
    all_coords = [coord for line in boundary_lines for coord in line[1]]
    centroid_x = sum(coord[0] for coord in all_coords) / len(all_coords)
    centroid_y = sum(coord[1] for coord in all_coords) / len(all_coords)

    # Sort by angle from centroid
    sorted_lines = sorted(boundary_lines, key=lambda x: atan2(x[1][0][1] - centroid_y, x[1][0][0] - centroid_x))

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

    min_distance = float('inf')
    closest_group_index = None

    for segment in boundary_lines:
        group_index, (start_point, end_point) = segment

        # Calculate distances from the origin to both start and end points of the segment
        distance_to_start = math.sqrt(start_point[0]**2 + start_point[1]**2)
        distance_to_end = math.sqrt(end_point[0]**2 + end_point[1]**2)

        # Check if either of the distances is the shortest found so far
        if distance_to_start < min_distance:
            min_distance = distance_to_start
            closest_group_index = group_index
        if distance_to_end < min_distance:
            min_distance = distance_to_end
            closest_group_index = group_index

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

    unit_index = []
    for i in range(len(unit_roads)):
        unit_index.append(unit_roads[i][0])

    if closest_group_index != 0:
        for i in range(len(unit_roads)):
            if unit_roads[i][0] == closest_group_index:
                unit_roads[i][0] = 0
            elif closest_group_index < unit_roads[i][0] <= max(unit_index):
                unit_roads[i][0] -= closest_group_index
            else:
                unit_roads[i][0] += (max(unit_index) - closest_group_index + 1)

    return unit_roads, closest_group_index