import os
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

from shapely.geometry import box

unit_length = 3.399999999942338e-05


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
        normalized_coords = [( (x - minx) / (maxx - minx), (y - miny) / (maxy - miny) ) for x, y in geometry.exterior.coords]
        return Polygon(normalized_coords)
    elif isinstance(geometry, LineString):
        normalized_coords = [( (x - minx) / (maxx - minx), (y - miny) / (maxy - miny) ) for x, y in geometry.coords]
        return LineString(normalized_coords)


from math import atan2, degrees

def create_rectangle(boundary_edge, farthest_point):
    x0, y0 = boundary_edge.coords[0]
    x1, y1 = boundary_edge.coords[-1]  # Use the last point of the combined edge
    xf, yf = farthest_point

    angle = degrees(atan2(y1 - y0, x1 - x0))

    if -45 <= angle <= 45 or 135 <= angle <= 225:  # Horizontal boundary
        # Calculate the coordinates of the rectangle's vertices
        x2, y2 = x1, yf
        x3, y3 = x0, yf
    else:  # Vertical boundary
        # Calculate the coordinates of the rectangle's vertices
        x2, y2 = xf, y1
        x3, y3 = xf, y0

    coords = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    return Polygon(coords)



def sorted_boundary_edges(boundary):
    edges = [(i, LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])) for i in range(len(boundary.exterior.coords) - 1)]
    # Sort by y-coordinate of the start point, then by x-coordinate
    sorted_edges = sorted(edges, key=lambda x: (x[1].coords[0][1], x[1].coords[0][0]))
    sorted_indices = [edge[0] for edge in sorted_edges]

    # Modify indices for edges shorter than the unit_length
    for i in range(1, len(sorted_indices)):
        b_edge = LineString([boundary.exterior.coords[sorted_indices[i]], boundary.exterior.coords[sorted_indices[i] + 1]])
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


def plot_groups_with_rectangles_v7(groups, boundary):
    updated_indices = updated_boundary_edge_indices_v5(boundary)
    unique_updated_indices = list(set(updated_indices))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_updated_indices)))
    group_colors = {idx: colors[i] for i, idx in enumerate(unique_updated_indices)}

    assigned_edges = set()

    for original_index in groups:
        edge_index = updated_indices[original_index]
        cluster_polygons = groups[original_index]
        farthest_polygon = find_polygon_with_farthest_edge(cluster_polygons, LineString([boundary.exterior.coords[original_index], boundary.exterior.coords[original_index + 1]]))

        mid_point = LineString(
            [boundary.exterior.coords[original_index], boundary.exterior.coords[original_index + 1]]).centroid
        plt.text(mid_point.x, mid_point.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')
        # Get all original indices that have the same updated index
        same_index_originals = [i for i, idx in enumerate(updated_indices) if idx == edge_index]

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

        same_index_originals = [i for i, idx in enumerate(updated_indices) if idx == edge_index]
        combined_edge = create_combined_edge(boundary, same_index_originals)
        rect_polygon = create_rectangle(combined_edge, farthest_point)

        # Draw the rectangle polygon
        rect_x, rect_y = rect_polygon.exterior.xy
        plt.plot(rect_x, rect_y, color=group_colors[edge_index], linestyle='--', linewidth=1, alpha=0.5)

        # Draw building polygon
        for poly in cluster_polygons:
            x, y = poly.exterior.xy
            plt.plot(x, y, color=group_colors[edge_index])
            centroid = poly.centroid
            plt.text(centroid.x, centroid.y, str(edge_index), fontsize=7, ha='center', va='center', color='black')

        # Draw the corresponding combined boundary edge with the same color
        x, y = combined_edge.xy
        plt.plot(x, y, color=group_colors[edge_index], linewidth=1)  # thicker line for combined boundary edge
        assigned_edges.update(same_index_originals)

    # Draw boundary edges that are not assigned to any group
    for i in range(len(boundary.exterior.coords) - 1):
        if i not in assigned_edges:
            x, y = zip(*[boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
            plt.plot(x, y, color='black', linewidth=1)
            mid_point = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]]).centroid
            plt.text(mid_point.x, mid_point.y, str(updated_indices[i]), fontsize=7, ha='center', va='center',
                     color='black')

        line_segment = LineString([boundary.exterior.coords[i], boundary.exterior.coords[i + 1]])
        unit_length_points = create_unit_length_points(line_segment, unit_length)
        for point in unit_length_points:
            plt.scatter(point.x, point.y, color='black', s=1)

    plt.show()


city_name = "portland"
dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
files = os.listdir(dir_path)
filenum = len(files)

building_polygons = []

for i in range(3, 5):
    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Combined_Buildings', f'{city_name}_buildings{i}.geojson')
    boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries', f'{city_name}_boundaries{i}.geojson')

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