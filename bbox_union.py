import os
import numpy as np
import geopandas as gpd
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

from shapely.geometry import box

unit_length = 0.04

import math


def compute_angle(p1, p2, p3):
    """Compute the internal angle at the middle point p2 of a triangle formed by points p1, p2, and p3."""
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

    # Step 3: Compute angle and Step 4: Update index if angle > 150
    adjustment = 0
    for i in range(1, len(merged_segments)):
        start1, end1 = merged_segments[i - 1]
        start2, end2 = merged_segments[i]

        # Compute angle between the two merged segments
        angle = compute_angle(boundary.exterior.coords[start1], boundary.exterior.coords[end1],
                              boundary.exterior.coords[end2])

        # Update index if angle > 150

        # print(angle)
        if angle > 100:
            for j in range(start2, end2):
                updated_indices[j] = updated_indices[start2 - 1]
                adjustment += 1
        else:
            for j in range(start2, end2):
                updated_indices[j] -= adjustment

    # print(updated_indices)

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


from math import atan2, degrees, sqrt


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
    edges = [(i, LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])) for i in
             range(len(boundary.exterior.coords) - 1)]
    # Sort by y-coordinate of the start point, then by x-coordinate
    sorted_edges = sorted(edges, key=lambda x: (x[1].coords[0][1], x[1].coords[0][0]))
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


def get_boundary_building_polygon_with_index(groups, boundary):
    updated_indices = updated_boundary_edge_indices_v7(boundary)
    unique_updated_indices = list(set(updated_indices))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_updated_indices)))
    group_colors = {idx: colors[i] for i, idx in enumerate(unique_updated_indices)}

    building_polygons = []
    boundary_lines = []
    unique_index = 0

    assigned_edges = set()

    for original_index in groups:
        edge_index = updated_indices[original_index]
        cluster_polygons = groups[original_index]

        same_index_originals = [i for i, idx in enumerate(updated_indices) if idx == edge_index]
        combined_edge = create_combined_edge(boundary, same_index_originals)
        # Draw building polygon
        for poly in cluster_polygons:
            unique_index += 1
            x, y = poly.exterior.xy
            # plt.plot(x, y, color=group_colors[edge_index])
            centroid = poly.centroid
            building_polygons.append([unique_index, edge_index, poly])
            # plt.text(centroid.x, centroid.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')

        # Draw the corresponding combined boundary edge with the same color
        x, y = combined_edge.xy

        boundary_lines.append([edge_index, ((x[0], y[0]), (x[1], y[1]))])
        # plt.plot(x, y, color=group_colors[edge_index], linewidth=1)

        assigned_edges.update(same_index_originals)

    for i in range(len(boundary.exterior.coords) - 1):
        if i not in assigned_edges:
            x, y = zip(*[boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
            boundary_lines.append([updated_indices[i], ((x[0], y[0]), (x[1], y[1]))])

            # plt.plot(x, y, color='black', linewidth=1)
            # mid_point = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]]).centroid
            # plt.text(mid_point.x, mid_point.y, str(updated_indices[i]), fontsize=7, ha='center', va='center',
            #          color='black')

    return building_polygons, boundary_lines


def plot_unit_segments(unit_segments, boundary):
    updated_indices = updated_boundary_edge_indices_v7(boundary)

    # Calculate the total number of unique indices based on the max value in updated_indices
    total_unique_indices = max(updated_indices) + 1
    colors = plt.cm.rainbow(np.linspace(0, 1, total_unique_indices))
    group_colors = {idx: colors[idx] for idx in range(total_unique_indices)}

    for street_index, segment in unit_segments:
        point1, point2 = segment
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]

        mid_pointx = (point1[0] + point2[0]) / 2
        mid_pointy = (point1[1] + point2[1]) / 2
        color = group_colors.get(street_index, 'black')  # Use black as default if no color is assigned
        plt.plot(x_values, y_values, color=color, linewidth=1)
        plt.text(mid_pointx, mid_pointy, str(street_index), fontsize=7, ha='center', va='center',
                 color='black')

    unique_index = 0

    for original_index in groups:
        edge_index = updated_indices[original_index]
        cluster_polygons = groups[original_index]
        for poly in cluster_polygons:
            unique_index += 1
            x, y = poly.exterior.xy
            plt.plot(x, y, color=group_colors[edge_index])
            centroid = poly.centroid
            building_text = str(edge_index)
            # plt.text(centroid.x, centroid.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')
            plt.text(centroid.x, centroid.y, building_text, fontsize=7, ha='center', va='center', color='black')

    plt.show()


def plot_groups_with_rectangles_v7(groups, boundary):
    updated_indices = updated_boundary_edge_indices_v5(boundary)
    unique_updated_indices = list(set(updated_indices))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_updated_indices)))
    group_colors = {idx: colors[i] for i, idx in enumerate(unique_updated_indices)}

    building_to_bbox_indices = defaultdict(set)
    assigned_edges = set()

    for original_index in groups:
        edge_index = updated_indices[original_index]
        cluster_polygons = groups[original_index]

        same_index_originals = [i for i, idx in enumerate(updated_indices) if idx == edge_index]
        combined_edge = create_combined_edge(boundary, same_index_originals)

        # Find the farthest polygon and point for the bounding box
        farthest_polygon = find_polygon_with_farthest_edge(cluster_polygons, combined_edge)
        farthest_point = None
        max_distance = 0
        for point in farthest_polygon.exterior.coords:
            distance = combined_edge.distance(Point(point))
            if distance > max_distance:
                max_distance = distance
                farthest_point = point

        rect_polygon = create_rectangle(combined_edge, farthest_point)

        # Check each building against the bounding box
        for i, building in enumerate(building_polygons):
            if rect_polygon.intersects(building):
                building_to_bbox_indices[i].add(edge_index)
                assigned_edges.add(original_index)

        # Visualization of Bounding Box
        rect_x, rect_y = rect_polygon.exterior.xy
        plt.plot(rect_x, rect_y, color=group_colors[edge_index], linestyle='--', linewidth=1, alpha=0.5)

        x, y = combined_edge.xy
        plt.plot(x, y, color=group_colors[edge_index], linewidth=1)
        mid_point = combined_edge.centroid
        plt.text(mid_point.x, mid_point.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')

        assigned_edges.update(same_index_originals)

        # Visualization of Buildings and Indices
    for i, building in enumerate(building_polygons):
        x, y = building.exterior.xy

        # Check if building is associated with any bounding box
        if building_to_bbox_indices[i]:
            # Use the color of the first bounding box index
            building_color = group_colors[next(iter(building_to_bbox_indices[i]))]
        else:
            # Default to black if not associated with any bounding box
            building_color = 'black'

        plt.plot(x, y, color=building_color)

        centroid = building.centroid
        building_text = ','.join(map(str, building_to_bbox_indices[i]))
        plt.text(centroid.x, centroid.y, building_text, fontsize=7, ha='center', va='center', color='black')

    # Draw boundary edges that are not assigned to any group
    for i in range(len(boundary.exterior.coords) - 1):
        if i not in assigned_edges:
            x, y = zip(*[boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
            plt.plot(x, y, color='black', linewidth=1)
            mid_point = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]]).centroid
            plt.text(mid_point.x, mid_point.y, str(updated_indices[i]), fontsize=7, ha='center', va='center', color='black')

        line_segment = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
        unit_length_points = create_unit_length_points(line_segment, unit_length)
        for point in unit_length_points:
            plt.scatter(point.x, point.y, color='black', s=1)

    plt.show()


city_name = "dublin"
dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Normalized','Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

building_polygons = []

for i in range(20, 100):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Normalized','Buildings', f'{city_name}_buildings{i}.geojson')
    boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                     'Normalized','Boundaries', f'{city_name}_boundaries{i}.geojson')

    if os.path.exists(building_filename):
        boundary_gdf = gpd.read_file(boundary_filename)
        building_gdf = gpd.read_file(building_filename)

        # Get building polygons for the current file and add them to the building_polygon list
        building_polygon = [row['geometry'] for idx, row in building_gdf.iterrows()]

        # Accumulate building polygons from all files into the building_polygons list
        building_polygons.extend(building_polygon)

        boundary_polygon = boundary_gdf.iloc[0]['geometry']

        sorted_edges = sorted_boundary_edges(boundary_polygon)
        groups = group_by_boundary_edge(building_polygon, boundary_polygon, sorted_edges)
        plot_groups_with_rectangles_v7(groups, boundary_polygon)
        building_polygons.clear()
        # building_polygons, boundary_lines = get_boundary_building_polygon_with_index(groups, boundary_polygon)

        # print(boundary_lines)

        # plot_unit_segments(boundary_lines, boundary_polygon)

        # print(building_polygons)
        # print(boundary_lines) # boundary street segment
        # print(boundary_polygon) # whole boundary
