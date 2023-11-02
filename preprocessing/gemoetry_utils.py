import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

def get_near_street_idx(building_polygon, boundary_lines):
    """
    Returns the index of the nearest boundary_line to the given building_polygon.
    """
    min_distance = float('inf')
    nearest_street_idx = -1

    for boundary_line in boundary_lines:
        # Iterate over each segment in the boundary line
        for segment in boundary_line[1]:
            # Convert the segment's coordinates to a shapely LineString
            line = LineString(segment)
            # Calculate the distance between the building_polygon and the line segment
            distance = building_polygon.distance(line)
            if distance < min_distance:
                min_distance = distance
                nearest_street_idx = boundary_line[0]

    return nearest_street_idx

def get_bbox_details(rotated_rectangle):
    # 사각형의 꼭짓점들을 얻음
    x, y = rotated_rectangle.exterior.coords.xy

    w = math.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    h = math.sqrt((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    theta = math.degrees(math.atan2(dy, dx))
    theta = theta / 180

    # 좌하단 꼭짓점을 x, y로 선택
    x, y = rotated_rectangle.centroid.x, rotated_rectangle.centroid.y

    return x, y, w, h, theta

def is_building_between(b1, b2, buildings):
    line = LineString([b1.centroid, b2.centroid])
    for building in buildings:
        if building[2].intersects(line) and building[2] != b1 and building[2] != b2:
            return True
    return False

def find_maximum_distance_from_polygon_to_linestring(polygon, linestring):
    max_distance = 0
    for coord in polygon.exterior.coords:
        point = Point(coord)
        point_distance = point.distance(linestring)
        max_distance = max(max_distance, point_distance)
    return max_distance


def find_nearest_linestring_for_each_polygon(polygons, linestrings):
    linestring_distances = {idx: [] for idx, _ in linestrings}

    for poly in polygons:
        min_distance = float('inf')
        nearest_line_index = None

        for idx, line in linestrings:
            distance = find_maximum_distance_from_polygon_to_linestring(poly, line)
            if distance < min_distance:
                min_distance = distance
                nearest_line_index = idx

        linestring_distances[nearest_line_index].append(min_distance)

    return linestring_distances


def find_maximum_distance_for_each_linestring(linestring_distances):
    max_distance_list = []
    for idx, distances in linestring_distances.items():
        if distances:  # Check if the list is not empty
            max_distance_list.append([idx, max(distances)])
    return max_distance_list


def distance_to_origin(point):
    """Calculate the distance between a point and the origin."""
    return point.distance(Point(0, 0))

def calculate_angle_from_reference(centroid, reference):
    """Calculate the angle between the centroid and a reference point."""
    dy = centroid.y - reference.y
    dx = centroid.x - reference.x
    angle = math.atan2(dy, dx)
    return angle

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


def create_rectangle(boundary_edge, farthest_point, linestring_list, edge_index):
    rect_edge = linestring_list[edge_index][1]
    # rect_edge = boundary_edge

    # Extract coordinates from boundary edge
    x0, y0 = boundary_edge.coords[0]
    x1, y1 = boundary_edge.coords[-1]

    # Calculate the direction vector of the boundary_edge
    dx = x1 - x0
    dy = y1 - y0

    # Calculate the length of the boundary_edge
    length_boundary_edge = sqrt(dx ** 2 + dy ** 2)

    x0, y0 = rect_edge.coords[0]
    x1, y1 = rect_edge.coords[-1]


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


    return Polygon(coords), distance_to_farthest_point


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

def angle_between(line1, line2):
    """Calculate the angle in degrees between two LineStrings."""
    # Extracting start and end coordinates of each LineString
    v1_start, v1_end = np.array(line1.coords[0]), np.array(line1.coords[-1])
    v2_start, v2_end = np.array(line2.coords[0]), np.array(line2.coords[-1])

    # Calculating direction vectors for each LineString
    v1 = v1_end - v1_start
    v2 = v2_end - v2_start

    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5
    cosine_angle = dot_product / magnitude_product
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


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
    line_direction = line_coords[-1] - line_coords[0]

    # List to store projected coordinates
    proj_coords = []

    for i in range(len(poly_coords) - 1):
        segment = LineString([poly_coords[i], poly_coords[i + 1]])
        if angle_between(segment, linestring) < 45:
            proj_coords.extend([project_point_onto_line(p, line_coords[0], line_coords[-1]) for p in [poly_coords[i], poly_coords[i+1]]])

    # Convert the list to a numpy array
    proj_coords = np.array(proj_coords)

    # Check if the projected polygon overlaps with the linestring
    projected_polygon = LineString(proj_coords)

    if projected_polygon.is_empty or linestring.is_empty:
        return False

    if projected_polygon.is_valid and linestring.is_valid:
        overlap = projected_polygon.intersects(linestring)
    else:
        overlap = False
    return overlap

