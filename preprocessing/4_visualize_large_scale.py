import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import math
import networkx as nx
from tqdm import tqdm

# Function to calculate the centroid of a polygon
def calculate_centroid(polygon):
    shapely_polygon = Polygon(polygon['coordinates'][0])
    centroid = shapely_polygon.centroid
    return centroid.x, centroid.y

# Function to restore predictions using scale_factor, rotation_angle, and centroid
def restore_predictions(predictions, scale_factor, rotation_angle, centroid):
    restored_predictions = []
    for pred in predictions:
        print(pred, rotation_angle)

        scaled_pos = np.array([pred[0] - 0.5, pred[1] - 0.5]) / scale_factor  # Scale position
        scaled_size = np.array(pred[2:4]) / scale_factor  # Scale size

        angle = -rotation_angle  # Rotate back by the negative angle

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        original_coords = np.dot(rotation_matrix, scaled_pos)
        original_coords = original_coords + centroid

        theta = np.deg2rad(pred[4:])  # Convert theta to radians
        theta = np.rad2deg([theta + angle]).tolist()

        restored_predictions.append(original_coords.tolist() + scaled_size.tolist() + theta)
    return restored_predictions

# Function to create a rotated rectangle polygon
def create_rotated_rectangle(x, y, w, h, theta):
    dx = w / 2
    dy = h / 2
    corners = [(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy)]

    theta = np.radians(theta)
    rotated_corners = [
        (math.cos(theta) * cx - math.sin(theta) * cy + x,
         math.sin(theta) * cx + math.cos(theta) * cy + y) for cx, cy in corners
    ]

    rotated_rectangle = Polygon(rotated_corners)
    return rotated_rectangle

# Load block building information and transformed block building information
block_building_info = pd.read_pickle('C:/Users/Dobby/Downloads/block_building_info.pkl')
transformed_block_building_info = pd.read_pickle('C:/Users/Dobby/Downloads/transformed_block_building_info.pkl')

# Predict file directory and file list
prediction_files_dir = 'C:/Users/Dobby/Downloads/synthetic_images_large_scale/cvae_graph_20240528_203924/'  # Example directory
prediction_files = [f for f in os.listdir(prediction_files_dir) if f.startswith('pred') and f.endswith('.pkl')]

# List to store all restored predictions
all_restored_predictions = []

# Process each prediction file
for prediction_file in prediction_files:
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
        all_restored_predictions.append(restored_prediction[0])

# Visualization of all predictions
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Draw restored building predictions
for pred in all_restored_predictions:
    x, y, w, h, theta = pred
    building_polygon = create_rotated_rectangle(x, y, w, h, theta)
    px, py = building_polygon.exterior.coords.xy
    facecolor = [120/256, 179/256, 125/256]
    ax.fill(px, py, edgecolor='black', facecolor=facecolor)

# Draw block boundaries
for block in block_building_info:
    block_polygon = Polygon(block['block_polygon']['coordinates'][0])
    bx, by = block_polygon.exterior.coords.xy
    ax.plot(bx, by, 'r-', linewidth=2)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-82.5014547871, -82.4846695231])
ax.set_ylim([27.967424055, 27.9740948558])
ax.set_axis_off()


# Save the plot as an image file for verification
plt.savefig('restored_predictions_map.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
