import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, polygonize, transform
import math
import random
from tqdm import tqdm
import pyproj
import folium

seed = 327
random.seed(seed)
np.random.seed(seed)

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

# Function to add a polygon to the map
def add_polygon(map_obj, polygon, color):
    folium.GeoJson(polygon, style_function=lambda x: {
        'color': f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, 1)',  # Convert to RGBA string
        'fillColor': f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, 0.9)',  # Convert to RGBA string
        'fillOpacity': 0.9
    }).add_to(map_obj)

# Define a function to transform geometries to a common coordinate system (UTM zone 19N for Rhode Island)
def transform_to_utm(geom):
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform
    return transform(project, geom)

def transform_to_wgs84(geom):
    project = pyproj.Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True).transform
    return transform(project, geom)

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

# Extract all coordinates from block polygons
all_x_coords = []
all_y_coords = []

for block in block_building_info:
    block_polygon = Polygon(block['block_polygon']['coordinates'][0])
    x_coords, y_coords = block_polygon.exterior.coords.xy
    all_x_coords.extend(x_coords)
    all_y_coords.extend(y_coords)

# Calculate min and max values for x and y coordinates
minx, maxx = min(all_x_coords), max(all_x_coords)
miny, maxy = min(all_y_coords), max(all_y_coords)

# Create a new folium map for visualization
north, south, east, west = 41.8075265791, 41.7963455875, -71.404870643, -71.425158872
midpoint = [(north + south) / 2, (east + west) / 2]
m = folium.Map(location=midpoint, zoom_start=16, tiles='Esri.WorldImagery')

# Dictionary to keep track of random colors for each block
block_colors = {}

# List to keep track of drawn building polygons
drawn_polygons = []

# Draw restored building predictions
for idx, pred in tqdm(all_restored_predictions, desc='Drawing predictions'):
    x, y, w, h, theta = pred
    building_polygon = create_rotated_rectangle(x, y, w, h, theta)

    # Get the block boundary
    block_polygon = Polygon(block_building_info[idx]['block_polygon']['coordinates'][0])

    # Check if the centroid overlaps with any existing building polygons
    if is_centroid_overlapping((x, y), drawn_polygons):
        continue

    n = 0
    # Check for overlap and if the building is within the block boundary
    while is_overlapping(building_polygon, drawn_polygons) or not block_polygon.contains(building_polygon):
        w *= 0.95  # Reduce width by 5%
        h *= 0.95  # Reduce height by 5%
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
    building_polygon_wgs84 = transform_to_wgs84(building_polygon)
    add_polygon(m, building_polygon_wgs84.__geo_interface__, facecolor)
    drawn_polygons.append(building_polygon)

# Save the map with loaded data as an HTML file
m.save('blocks_with_buildings_loaded_map.html')
