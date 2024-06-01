import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Load the pickle file
file_path = 'block_building_info.pkl'
data = pd.read_pickle(file_path)

# Helper function to plot a polygon
def plot_polygon(ax, polygon, edge_color='blue', face_color='none'):
    coords = polygon['coordinates'][0]
    poly = Polygon(coords)
    patch = patches.Polygon(list(poly.exterior.coords), closed=True, edgecolor=edge_color, facecolor=face_color)
    ax.add_patch(patch)

# Function to calculate the centroid of a polygon
def calculate_centroid(polygon):
    coords = polygon['coordinates'][0]
    poly = Polygon(coords)
    return poly.centroid

# Function to calculate the angle to rotate the longest side to be horizontal
def calculate_rotation_angle(polygon):
    coords = polygon['coordinates'][0]
    poly = Polygon(coords)
    min_x, min_y, max_x, max_y = poly.bounds
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    if bbox_width >= bbox_height:
        return 0  # No rotation needed
    else:
        return np.pi / 2  # 90 degrees rotation

# Function to rotate a polygon
def rotate_polygon(polygon, angle, center_x, center_y):
    coords = polygon['coordinates'][0]
    rotated_coords = []
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    for x, y in coords:
        tx = x - center_x
        ty = y - center_y
        rot_x = tx * cos_angle - ty * sin_angle + center_x
        rot_y = tx * sin_angle + ty * cos_angle + center_y
        rotated_coords.append((rot_x, rot_y))
    return {'coordinates': [rotated_coords]}

# Function to normalize polygons
def normalize_polygon(polygon, scale_factor, center_x, center_y):
    coords = polygon['coordinates'][0]
    normalized_coords = []
    for x, y in coords:
        norm_x = (x - center_x) * scale_factor + 0.5  # Shift by 0.5 for center
        norm_y = (y - center_y) * scale_factor + 0.5  # Shift by 0.5 for center
        normalized_coords.append((norm_x, norm_y))
    return {'coordinates': [normalized_coords]}

# List to hold transformed data
transformed_data = []

# Create PNGs for each block
for idx, block_info in enumerate(data):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Calculate the centroid of the block
    centroid = calculate_centroid(block_info['block_polygon'])
    center_x, center_y = centroid.x, centroid.y

    # Calculate the rotation angle
    rotation_angle = calculate_rotation_angle(block_info['block_polygon'])

    # Rotate the block polygon
    rotated_block = rotate_polygon(block_info['block_polygon'], rotation_angle, center_x, center_y)

    # Calculate the bounding box of the rotated block
    block_coords = rotated_block['coordinates'][0]
    block_poly = Polygon(block_coords)
    min_x, min_y, max_x, max_y = block_poly.bounds

    # Determine the scale factor to normalize the longest side to 1
    max_length = max(max_x - min_x, max_y - min_y)
    scale_factor = 1 / max_length

    # Normalize the rotated block polygon
    normalized_block = normalize_polygon(rotated_block, scale_factor, center_x, center_y)
    plot_polygon(ax, normalized_block, edge_color='black')

    # Normalize and plot each building bounding box
    normalized_buildings = []
    for bbox in block_info['buildings_bbox']:
        rotated_bbox = rotate_polygon(bbox, rotation_angle, center_x, center_y)
        normalized_bbox = normalize_polygon(rotated_bbox, scale_factor, center_x, center_y)
        normalized_buildings.append(normalized_bbox)
        plot_polygon(ax, normalized_bbox, edge_color='red')

    # Add the transformed block and buildings to the list
    transformed_data.append({
        'block_polygon': normalized_block,
        'buildings_bbox': normalized_buildings,
        'scale_factor': scale_factor
    })

    # Set limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Block {idx + 1}')
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')

    # Save the figure
    plt.savefig(f'block_{idx + 1}.png')
    plt.close()

# Save transformed data to a new pickle file
transformed_data_df = pd.DataFrame(transformed_data)
transformed_data_df.to_pickle('transformed_block_building_info.pkl')
