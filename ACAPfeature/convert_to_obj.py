import numpy as np
import pandas as pd
import os
from scipy.spatial import Delaunay
import pickle

pointnum = 16

city_list = ['lasvegas', 'littlerock', 'manchester', 'milan', 'minneapolis', 'nottingham', 'paris', 'philadelphia', 'phoenix', 'portland', 'richmond', 'saintpaul', 'sanfrancisco']

def convert_csv_to_obj_using_pattern(csv_filename, output_filename, triangulation_pattern=None):
    # Load data from CSV
    data = pd.read_csv(csv_filename, header=None, names=["x", "y"])

    # Compute Delaunay triangulation if no pattern is given
    if triangulation_pattern is None:
        tri = Delaunay(data.values)
        simplices = tri.simplices
    else:
        simplices = triangulation_pattern

    with open(output_filename, 'w') as f:
        # Write vertices
        for _, row in data.iterrows():
            f.write(f"v {row['x']} {row['y']} 0\n")

        # Write faces using the given triangulation pattern
        for simplex in simplices:
            f.write(f"f {simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}\n")

    return simplices

def convert_pickle_to_obj_using_pattern(pickle_filename, output_filename, triangulation_pattern=None):
    # Load data from pickle
    with open(pickle_filename, 'rb') as f:
        points = pickle.load(f)

    # Convert points to DataFrame for consistency with the original code
    data = pd.DataFrame(points, columns=["x", "y"])

    # Compute Delaunay triangulation if no pattern is given
    if triangulation_pattern is None:
        tri = Delaunay(data.values)
        simplices = tri.simplices
    else:
        simplices = triangulation_pattern

    with open(output_filename, 'w') as f:
        # Write vertices
        for _, row in data.iterrows():
            f.write(f"v {row['x']} {row['y']} 0\n")

        # Write faces using the given triangulation pattern
        for simplex in simplices:
            f.write(f"f {simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}\n")

    return simplices


def is_point_inside_triangle(pt, tri):
    """Check if point 'pt' is inside triangle 'tri'."""

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, tri[0], tri[1])
    d2 = sign(pt, tri[1], tri[2])
    d3 = sign(pt, tri[2], tri[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def is_ear(p, v, q, polygon):
    """Check if the vertex 'v' is an ear."""
    if not is_clockwise(p, v, q):
        return False
    tri = (p, v, q)
    for x in polygon:
        if x in tri:
            continue
        if is_point_inside_triangle(x, tri):
            return False
    return True

def is_clockwise(p, v, q):
    """Check if the sequence of points is clockwise."""
    return (q[1] - p[1]) * (v[0] - q[0]) > (v[1] - q[1]) * (q[0] - p[0])

def ear_clipping_triangulation(polygon):
    """Perform ear clipping triangulation on a given polygon."""
    if len(polygon) < 3:
        return []

    triangles = []
    polygon = polygon[:]
    while len(polygon) > 3:
        found_ear = False
        for i in range(len(polygon)):
            p, v, q = polygon[i-1], polygon[i], polygon[(i+1)%len(polygon)]
            if is_ear(p, v, q, polygon):
                triangles.append((p, v, q))
                del polygon[i]
                found_ear = True
                break
        if not found_ear:
            break
    if len(polygon) == 3:
        triangles.append(tuple(polygon))
    return triangles

def save_triangulation_to_obj(triangulation, output_filename):
    vertices = list(set([vertex for triangle in triangulation for vertex in triangle]))
    vertex_to_index = {vertex: i for i, vertex in enumerate(vertices)}

    with open(output_filename, 'w') as f:
        # Write vertices to the .obj file
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} 0\n")

        # Write faces to the .obj file
        for triangle in triangulation:
            f.write(f"f {vertex_to_index[triangle[0]] + 1} {vertex_to_index[triangle[1]] + 1} {vertex_to_index[triangle[2]] + 1}\n")

def convert_pickle_polygon_to_obj(pickle_filename, output_filename):
    # Load the 2D polygon coordinates from the pickle file
    with open(pickle_filename, 'rb') as f:
        polygon_coords = pickle.load(f)

    # Perform ear clipping triangulation on the loaded polygon
    triangles = ear_clipping_triangulation(polygon_coords)

    # Save the triangulation to an .obj file
    save_triangulation_to_obj(triangles, output_filename)

def save_polygon_to_obj(points, filename):
    with open(filename, 'w') as file:
        # Write vertices (excluding the last point)
        for point in points[:-1]:
            file.write(f"v {point[0]} {point[1]} 0\n")
        # Write face
        file.write("f " + " ".join([str(i+1) for i in range(len(points)-1)]))


def normalize_polygon(vertices):
    min_values = [min(vertices, key=lambda v: v[i])[i] for i in range(2)]
    max_values = [max(vertices, key=lambda v: v[i])[i] for i in range(2)]

    normalized_vertices = []
    for vertex in vertices:
        normalized_vertex = [(vertex[i] - min_values[i]) / (max_values[i] - min_values[i]) for i in range(2)]
        normalized_vertices.append(normalized_vertex)

    return normalized_vertices


if __name__ == "__main__":
    sample_polygon_root_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "5_ACAP")
    pickle_path = os.path.join(sample_polygon_root_path, "unitsquare_596.pkl")
    obj_path = os.path.join(sample_polygon_root_path, "unitsquare_596.obj")

    convert_pickle_to_obj_using_pattern(pickle_path, obj_path)

    # with open(pickle_path, 'rb') as file:
    #     points = pickle.load(file)
    #
    # normalized_polygon = normalize_polygon(points.exterior.coords)
    #
    # print(normalized_polygon)
    #
    # save_polygon_to_obj(normalized_polygon, obj_path)
    # square_path = os.path.join(sample_polygon_root_path, "unitsquare_16.pkl")
    # obj_path = os.path.join(sample_polygon_root_path, "unitsquare_16.obj")

    # convert_pickle_to_obj_using_pattern(square_path, obj_path)

    # # Convert normalized_points.csv to square.obj and get its triangulation pattern
    # for idx in range(len(city_list)):
    #     city_name = city_list[idx]
    #
    #     square_csv = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'square_{pointnum}.csv')
    #     square_obj = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'square_{pointnum}.obj')
    #
    #     triangulation_pattern = convert_csv_to_obj_using_pattern(square_csv, square_obj)
    #
    #     # Directory containing the csv files you want to convert
    #     csv_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'Boundarycsv_{pointnum}')
    #
    #     # Directory where you want to save the .obj files
    #     obj_directory = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaryobj')
    #
    #     # Make the obj_directory if it doesn't exist
    #     if not os.path.exists(obj_directory):
    #         os.makedirs(obj_directory)
    #
    #     # Get list of all CSV files in the register directory
    #     all_csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    #
    #     for csv_file in all_csv_files:
    #         input_path = os.path.join(csv_directory, csv_file)
    #         # Construct the name for the output .obj file based on the csv file name
    #         output_filename = csv_file.replace('.csv', '.obj')
    #         output_path = os.path.join(obj_directory, output_filename)
    #         # Convert current CSV to .obj using the triangulation pattern
    #         convert_csv_to_obj_using_pattern(input_path, output_path, triangulation_pattern)
    #
    #     print(f"{city_name} done")