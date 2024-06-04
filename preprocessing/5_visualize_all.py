import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import math
import random
from tqdm import tqdm
import pyproj
import pickle
import geopandas as gpd

# Define the coordinates for the bounding box
north, south, east, west = 41.8075265791, 41.7963455875, -71.404870643, -71.425158872

# Create the bounding box
bbox = (north, south, east, west)

# Define a function to transform geometries to a common coordinate system (UTM zone 19N for Rhode Island)
def transform_to_utm(geom):
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform
    return transform(project, geom)

def transform_to_wgs84(geom):
    project = pyproj.Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True).transform
    return transform(project, geom)

# Load other elements
with open('natural_elements_utm.pkl', 'rb') as f:
    other_elements_utm = pickle.load(f)

# Function to assign colors based on tags
def get_color_by_tag(row):
    if row.get('landuse') == 'forest':
        return 'darkgreen'
    elif row.get('landuse') == 'grass':
        return 'limegreen'
    elif row.get('landuse') == 'meadow':
        return 'yellowgreen'
    elif row.get('landuse') == 'park':
        return 'blue'
    elif row.get('natural') in ['wood', 'scrub', 'heath']:
        return 'forestgreen'
    elif row.get('natural') in ['grassland']:
        return 'greenyellow'
    elif row.get('natural') in ['wetland']:
        return 'blue'
    elif row.get('natural') in ['water', 'bay']:
        return 'aqua'
    elif row.get('natural') in ['cape', 'beach']:
        return 'sandybrown'
    elif row.get('leisure') == 'park':
        return 'blue'
    elif row.get('leisure') == 'playground':
        return 'pink'
    elif row.get('leisure') == 'garden':
        return 'violet'
    elif row.get('leisure') == 'pitch':
        return 'orange'
    return 'gray'  # Default color

# Define a function to get category label
def get_category_label(row):
    if row.get('landuse') == 'forest':
        return 'Forest'
    elif row.get('landuse') == 'grass':
        return 'Grass'
    elif row.get('landuse') == 'meadow':
        return 'Meadow'
    elif row.get('landuse') == 'park':
        return 'Park'
    elif row.get('natural') in ['wood', 'scrub', 'heath']:
        return 'Wood/Scrub/Heath'
    elif row.get('natural') in ['grassland']:
        return 'Grassland'
    elif row.get('natural') in ['wetland']:
        return 'Wetland'
    elif row.get('natural') in ['water', 'bay']:
        return 'Water/Bay'
    elif row.get('natural') in ['cape', 'beach']:
        return 'Cape/Beach'
    elif row.get('leisure') == 'park':
        return 'Park'
    elif row.get('leisure') == 'playground':
        return 'Playground'
    elif row.get('leisure') == 'garden':
        return 'Garden'
    elif row.get('leisure') == 'pitch':
        return 'Pitch'
    return 'Other'  # Default label

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

# 저장된 예측값을 파일로 저장
with open('restored_predictions.pkl', 'wb') as f:
    pickle.dump(all_restored_predictions, f)

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

# Visualization of all elements and predictions
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

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
    px, py = building_polygon.exterior.coords.xy
    ax.fill(px, py, edgecolor='black', facecolor=facecolor)
    drawn_polygons.append(building_polygon)

# Plot other elements with different colors based on tags
if isinstance(other_elements_utm, gpd.GeoDataFrame):
    for _, row in other_elements_utm.iterrows():
        geometry = row['geometry']
        color = get_color_by_tag(row)
        if geometry.geom_type == 'Polygon':
            x, y = geometry.exterior.xy
            if color != 'gray':
                ax.fill(x, y, color=color, alpha=0.5)
            # Label the category
            centroid = geometry.centroid
            ax.text(centroid.x, centroid.y, get_category_label(row), fontsize=8, ha='center')

# Draw block boundaries
for block in block_building_info:
    block_polygon = Polygon(block['block_polygon']['coordinates'][0])
    bx, by = block_polygon.exterior.coords.xy
    ax.plot(bx, by, 'gray', linewidth=2)

# Add a legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='darkgreen', lw=4, label='Forest'),
    Line2D([0], [0], color='limegreen', lw=4, label='Grass'),
    Line2D([0], [0], color='yellowgreen', lw=4, label='Meadow'),
    Line2D([0], [0], color='blue', lw=4, label='Park'),
    Line2D([0], [0], color='forestgreen', lw=4, label='Wood/Scrub/Heath'),
    Line2D([0], [0], color='greenyellow', lw=4, label='Grassland'),
    Line2D([0], [0], color='blue', lw=4, label='Wetland'),
    Line2D([0], [0], color='aqua', lw=4, label='Water/Bay'),
    Line2D([0], [0], color='sandybrown', lw=4, label='Cape/Beach'),
    Line2D([0], [0], color='pink', lw=4, label='Playground'),
    Line2D([0], [0], color='violet', lw=4, label='Garden'),
    Line2D([0], [0], color='orange', lw=4, label='Pitch'),
    Line2D([0], [0], color='gray', lw=4, label='Other')
]
ax.legend(handles=legend_elements, loc='upper right')

# Set dynamic limits for the plot
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_aspect('equal', adjustable='box')
ax.set_axis_off()

# Save the plot as an image file for verification
plt.savefig('restored_predictions_with_elements_map.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
