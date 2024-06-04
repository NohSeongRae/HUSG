import bpy
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import math
import random
from tqdm import tqdm
from scipy.spatial import Delaunay


def normalize_coordinates(building_polygon, min_x, min_y, max_x, max_y, target_min=-10, target_max=10):
    # Get the exterior coordinates of the polygon
    x_coords, y_coords = zip(*[(vx, vy) for vx, vy in building_polygon.exterior.coords])

    # Define the normalization function
    def normalize(value, min_val, max_val, target_min, target_max):
        return ((value - min_val) / (max_val - min_val)) * (target_max - target_min) + target_min

    # Normalize the coordinates
    normalized_coords = [(normalize(vx, min_x, max_x, target_min, target_max),
                          normalize(vy, min_y, max_y, target_min, target_max) * (max_y - min_y) / (max_x - min_x))
                         for vx, vy in zip(x_coords, y_coords)]
    return normalized_coords


def normalize_polygon(building_polygon, min_x, min_y, max_x, max_y, target_min=-10, target_max=10):
    # Normalize the coordinates of the building polygon
    normalized_coords = normalize_coordinates(building_polygon, min_x, min_y, max_x, max_y, target_min, target_max)

    # Create a new normalized polygon
    normalized_polygon = Polygon(normalized_coords)

    return normalized_polygon

# Function to calculate the centroid of a polygon
def calculate_centroid(polygon):
    shapely_polygon = Polygon(polygon['coordinates'][0])
    centroid = shapely_polygon.centroid
    return centroid.x, centroid.y


# Function to restore predictions using scale_factor, rotation_angle, and centroid
def restore_predictions(predictions, scale_factor, rotation_angle, centroid):
    restored_predictions = []
    for pred in predictions:
        scaled_pos = np.array([pred[0] - 0.5, pred[1] - 0.5]) / scale_factor  # Scale position
        scaled_size = np.array(pred[2:4]) / scale_factor  # Scale size

        angle = -rotation_angle  # Rotate back by the negative angle

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        original_coords = np.dot(rotation_matrix, scaled_pos)
        original_coords = original_coords + centroid

        theta = np.deg2rad(pred[4:])  # Convert theta to radians
        theta = np.rad2deg([theta - angle]).tolist()

        restored_predictions.append(original_coords.tolist() + scaled_size.tolist() + theta)
    return restored_predictions


# Function to create a rotated rectangle polygon
def create_rotated_rectangle(x, y, w, h, theta):
    dx = w / 2
    dy = h / 2
    corners = [(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy)]

    theta = -np.radians(theta)
    rotated_corners = [
        (math.cos(theta) * cx - math.sin(theta) * cy + x,
         math.sin(theta) * cx + math.cos(theta) * cy + y) for cx, cy in corners
    ]

    rotated_rectangle = Polygon(rotated_corners)
    return rotated_rectangle


# Function to check if a polygon overlaps with any in a list of polygons
def is_overlapping(polygon, polygon_list):
    for poly in polygon_list:
        if polygon.intersects(poly):
            return True
    return False


# Function to check if a point overlaps with any in a list of polygons
def is_centroid_overlapping(centroid, polygon_list):
    point = Point(centroid)
    for poly in polygon_list:
        if poly.contains(point):
            return True
    return False


# Function to generate a pastel color
def generate_pastel_color():
    base_color = np.array([random.random(), random.random(), random.random()])
    white = np.array([1.0, 1.0, 1.0])
    pastel_color = (base_color + white) / 2
    return pastel_color


# Load block building information and transformed block building information
block_building_info = pd.read_pickle('C:/Users/Dobby/Downloads/block_building_info.pkl')
transformed_block_building_info = pd.read_pickle('C:/Users/Dobby/Downloads/transformed_block_building_info.pkl')

# Predict file directory and file list
prediction_files_dir = 'C:/Users/Dobby/Downloads/gt_images_large_scale/cvae_graph_20240528_203924/'  # Example directory
prediction_files = [f for f in os.listdir(prediction_files_dir) if f.startswith('pred') and f.endswith('.pkl')]

# List to store all restored predictions
all_restored_predictions = []

# Process each prediction file
for prediction_file in tqdm(prediction_files, desc='Processing prediction files'):
    idx = int(prediction_file.split('_')[-1].split('.')[0])
    # Load prediction file
    predictions = pd.read_pickle(os.path.join(prediction_files_dir, prediction_file))

    # Calculate centroids for each block
    centroids = [calculate_centroid(block['block_polygon']) for block in block_building_info]

    # Restore predictions for each file
    for prediction in predictions:
        scale_factor = transformed_block_building_info.iloc[idx]['scale_factor']
        rotation_angle = transformed_block_building_info.iloc[idx]['rotation_angle']
        centroid = centroids[idx]
        restored_prediction = restore_predictions([prediction], scale_factor, rotation_angle, centroid)
        all_restored_predictions.append((idx, restored_prediction[0]))


# Function to create Blender mesh from a list of vertices
def create_mesh(name, vertices, faces, color):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)

    if len(color) != 0:
        red = color[0]
        green = color[1]
        blue = color[2]
        alpha = 1.0
        color = (red, green, blue, alpha)

        material = bpy.data.materials.new("random_material")
        material.diffuse_color = color

        obj.data.materials.append(material)

    bpy.context.collection.objects.link(obj)
    return obj


# Function to generate a random height for buildings using a Gaussian distribution
def generate_random_height(mean=2.5, std_dev=0.3, lower_bound=2, upper_bound=3):
    while True:
        height = np.random.normal(mean, std_dev)
        if lower_bound <= height <= upper_bound:
            return height / 4


# Dictionary to keep track of random colors for each block
block_colors = {}

# Create and draw restored building predictions
drawn_polygons = []
mean_x, mean_y = 0, 0
for idx, pred in all_restored_predictions:
    mean_x += pred[0]
    mean_y += pred[1]
mean_x /= len(all_restored_predictions)
mean_y /= len(all_restored_predictions)

min_x, min_y = float('inf'), float('inf')
max_x, max_y = float('-inf'), float('-inf')

for idx, pred in all_restored_predictions:
    min_x = min(min_x, pred[0])
    min_y = min(min_y, pred[1])
    max_x = max(max_x, pred[0])
    max_y = max(max_y, pred[1])

for idx, pred in tqdm(all_restored_predictions, desc='Drawing predictions'):
    x, y, w, h, theta = pred
    building_polygon = create_rotated_rectangle(x, y, w, h, theta)

    # 블록 폴리곤 좌표 가져오기
    block_polygon_coords = block_building_info[idx]['block_polygon']['coordinates'][0]
    block_polygon = Polygon(block_polygon_coords)

    if is_centroid_overlapping((x, y), drawn_polygons):
        continue

    n = 0
    while is_overlapping(building_polygon, drawn_polygons) or not block_polygon.contains(building_polygon):
        w *= 0.95
        h *= 0.95
        building_polygon = create_rotated_rectangle(x, y, w, h, theta)

        n += 1
        if n > 10:
            n = -1
            break

    if n == -1:
        continue

    if idx not in block_colors:
        block_colors[idx] = generate_pastel_color()

    facecolor = block_colors[idx]
    drawn_polygons.append(building_polygon)

    building_polygon = normalize_polygon(building_polygon, min_x, min_y, max_x, max_y, target_min=-10, target_max=10)

    vertices = [(vx, vy, 0) for vx, vy in building_polygon.exterior.coords]
    faces = [(i, (i + 1) % len(vertices), (i + 2) % len(vertices), (i + 3) % len(vertices)) for i in
             range(len(vertices) - 1)]
    bottom_face = create_mesh(f'building_{idx}', vertices, faces, facecolor)

    # Generate random height and create building volume
    height = generate_random_height()
    top_vertices = [(vx, vy, height) for vx, vy in building_polygon.exterior.coords]

    # Create side faces
    side_faces = []
    for i in range(len(vertices)):
        next_i = (i + 1) % len(vertices)
        side_faces.append((i, next_i, next_i + len(vertices), i + len(vertices)))

    vertices += top_vertices
    faces += [(i + len(building_polygon.exterior.coords),
               (i + 1) % len(building_polygon.exterior.coords) + len(building_polygon.exterior.coords),
               (i + 2) % len(building_polygon.exterior.coords) + len(building_polygon.exterior.coords),
               (i + 3) % len(building_polygon.exterior.coords) + len(building_polygon.exterior.coords)) for i in
              range(len(building_polygon.exterior.coords) - 1)]
    faces += side_faces
    create_mesh(f'building_volume_{idx}', vertices, faces, facecolor)


def create_delaunay_mesh(polygon):
    # 다각형의 외부 좌표 가져오기
    exterior_coords = np.array(polygon.exterior.coords)

    # Delaunay 삼각 분할 생성
    tri = Delaunay(exterior_coords)

    # 정점 생성 (Z 좌표를 0으로 설정)
    vertices = [(vx, vy, 0) for vx, vy in exterior_coords]

    # 삼각형 얼굴 생성
    faces = [(triangle[0], triangle[1], triangle[2], triangle[0]) for triangle in tri.simplices ]

    new_faces = []
    for idx, triangle in enumerate(tri.simplices):
        face_polygon = Polygon([vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]])
        if face_polygon.intersects(polygon):
            if (face_polygon.intersection(polygon).area/face_polygon.area)*100 > 50:
                new_faces.append((triangle[0], triangle[1], triangle[2], triangle[0]))

    return vertices, new_faces

# Draw block boundaries
for block in block_building_info:
    block_polygon = Polygon(block['block_polygon']['coordinates'][0])
    block_polygon = normalize_polygon(block_polygon, min_x, min_y, max_x, max_y, target_min=-10, target_max=10)

    vertices, faces = create_delaunay_mesh(block_polygon)

    create_mesh(f'block_boundary_{block_building_info.index(block)}', vertices, faces, [1, 1, 1])

# Save the Blender file
bpy.ops.wm.save_as_mainfile(filepath='C:/Users/Dobby/Downloads/restored_buildings.blend')