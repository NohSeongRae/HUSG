import os
from shapely.geometry import Polygon, MultiPoint, Point, LineString, shape, LinearRing
from shapely.ops import nearest_points
import numpy as np
import matplotlib.pyplot as plt
import json
import geopandas
from tqdm import tqdm
import matplotlib
import pickle
matplotlib.use('TkAgg')

path=r'C:\Users\rhosu\Desktop\ssss\example.pkl'

# Function to compute the Hausdorff distance between two polygons using MultiPoint for efficiency
def hausdorff_distance(polygon1, polygon2):
    return MultiPoint(polygon1).hausdorff_distance(MultiPoint(polygon2))


def add_vertices_to_polygon(polygon, target_num_vertices):
    points = list(polygon.exterior.coords[:-1])  # Assuming closed polygon, ignore last point since it's a duplicate

    while len(points) < target_num_vertices:
        # Find the longest edge
        edge_lengths = [LineString([points[i], points[(i + 1) % len(points)]]).length for i in range(len(points))]
        longest_edge_index = edge_lengths.index(max(edge_lengths))
        # Split the longest edge by adding a midpoint
        p1, p2 = points[longest_edge_index], points[(longest_edge_index + 1) % len(points)]
        midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        # Insert the midpoint
        points.insert(longest_edge_index + 1, midpoint)

    # After all vertices are added, check for duplicates and adjust if necessary
    # We need to check in a circular manner, meaning the last point connects back to the first
    for i in range(len(points)):
        # Check if there's a duplicate of the current point
        if points.count(points[i]) > 1:
            # Find the next unique point to calculate the midpoint
            for j in range(1, len(points)):
                next_index = (i + j) % len(points)
                if points[next_index] != points[i]:
                    # Found the next unique point, calculate the midpoint
                    midpoint = ((points[i][0] + points[next_index][0]) / 2, (points[i][1] + points[next_index][1]) / 2)
                    points[i] = midpoint  # Move the current point to the midpoint
                    break

    # Close the polygon by adding the first point at the end
    points.append(points[0])
    if len(points) != len(set(points)) + 1 :
        print("Duplicate points found:", points)
    # Create and return the new polygon
    new_polygon = Polygon(points)
    return new_polygon


def process_polygon(polygon, target_num_vertices=64):
    num_vertices = len(polygon.exterior.coords) - 1  # excluding the closing point which is a duplicate
    if num_vertices < target_num_vertices:
        # Add vertices
        print('under vertices threshold')
        return add_vertices_to_polygon(polygon, target_num_vertices)
    elif num_vertices == target_num_vertices:
        # Do nothing, just return the polygon
        print('vertices threshold')
        return polygon
    else:
        print('over vertices threshold')
        # Simplify the polygon by removing vertices
        simplified_polygon = remove_vertices_optimized(polygon, target_num_vertices)
        num_vertices_after_simplification = len(simplified_polygon.exterior.coords) - 1

        if num_vertices_after_simplification < target_num_vertices:
            # Add vertices to make it exactly target_num_vertices
            return add_vertices_to_polygon(simplified_polygon, target_num_vertices)
        elif num_vertices_after_simplification == target_num_vertices:
            # If we end up with the target number, we make it a rectangle
            return simplified_polygon
        else:
            raise ValueError("The simplification process did not reduce the vertices as expected.")

def remove_vertices_optimized(target_polygon, num_vertices=64):
    points = list(target_polygon.exterior.coords)[:-1]  # exclude the closing point which is a duplicate of the first
    while len(points) > num_vertices:
        min_distance_increase = float('inf')
        vertex_to_remove = -1
        for i in range(len(points)):
            new_points = points[:i] + points[i+1:]
            new_polygon = Polygon(new_points)
            distance_increase = hausdorff_distance(target_polygon.exterior.coords, new_points)
            if distance_increase < min_distance_increase:
                min_distance_increase = distance_increase
                vertex_to_remove = i
        del points[vertex_to_remove]
    return Polygon(points)
def geojson_to_shapely_polygon(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Assuming the GeoJSON contains only one feature (a polygon)
    geometry = data['features'][0]['geometry']
    polygon = shape(geometry)

    return polygon


building_file_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "atlanta_dataset",
                                  "density20_building120_rotate_normalized", "Buildings",
                                  "atlanta_buildings295.geojson")
target_polygon = geojson_to_shapely_polygon(building_file_path)


# # Mock polygon with more than 8 vertices (for demonstration purposes)
# original_polygon_coords = [(0, 0), (2, 1), (4, 0), (3, 3), (4, 4), (3, 5), (4, 6), (3, 7), (1, 7), (0, 6), (1, 5), (0, 4), (1, 3), (0, 0)]
# original_polygon = Polygon(original_polygon_coords)

# Run the optimized version of the vertex removal
simplified_polygon_optimized = process_polygon(target_polygon)

original_polygon_coords = list(target_polygon.exterior.coords)
simplified_polygon_coords = list(simplified_polygon_optimized.exterior.coords)
# Check the result
plt.figure(figsize=(8, 8))
plt.plot(*zip(*original_polygon_coords), label='Original Polygon', color='blue')
plt.plot(*zip(*simplified_polygon_coords), label='Simplified Polygon', color='red', alpha=0.5)
plt.fill(*zip(*simplified_polygon_coords), color='red', alpha=0.2)
for x, y in simplified_polygon_coords:
    plt.plot(x, y, marker='o', markersize=5, color='green')
plt.title("Original vs Simplified Polygon")
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()