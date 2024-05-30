import os
from tqdm import tqdm
import numpy as np
import pickle
from shapely.geometry import Polygon
import math
import matplotlib.pyplot as plt
import networkx as nx

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

path = "C:/Users/Dobby/Downloads/transformer_ablation/synthetic_images_epoch110/cvae_graph_20240528_203924"
model = path.split('/')[-1]

list_output_all = os.listdir(path)
list_output = [name for name in list_output_all if 'prediction' in name and 'pkl' in name]

for output in tqdm(list_output):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    file_path = f'{path}/{output}'
    with open(file_path, 'rb') as f:
        tmp = pickle.load(f)

    for t in tmp:
        building_polygon = create_rotated_rectangle(t[0], t[1], t[2], t[3], t[4])
        x, y = building_polygon.exterior.coords.xy
        facecolor = [120, 179, 125]
        for i in range(3):
            facecolor[i] /= 256
        ax.fill(x, y, edgecolor='black', facecolor=facecolor)

    file_idx = file_path.split('/')[-1].replace('prediction_', '').replace('.pkl', '')

    boundary_path = f'C:/Users/Dobby/Downloads/graph_condition_city_datasets/ours_city_datasets/graph_condition_train_datasets/test/{file_idx}.gpickle'
    graph = nx.read_gpickle(boundary_path)

    boundary_points = []
    for node in graph.graph['condition'].nodes():
        x_min, y_min, x_max, y_max, theta = graph.graph['condition'].nodes[node]['chunk_features']
        boundary_polygon = create_rotated_rectangle(x_min, y_min, x_max, y_max, (theta * 2 - 1) * 45)
        x, y = boundary_polygon.exterior.coords.xy
        color = [89, 89, 89]
        for i in range(3):
            color[i] /= 256
        ax.plot(x, y, '-', color=color)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

    directory = path.replace(model, 'figure_pred_all')
    os.makedirs(directory, exist_ok=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_axis_off()
    save_path_1 = os.path.join(directory, file_idx + ".png")
    ax.figure.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close(ax.figure)

    # Boundary only plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for node in graph.graph['condition'].nodes():
        x_min, y_min, x_max, y_max, theta = graph.graph['condition'].nodes[node]['chunk_features']
        boundary_polygon = create_rotated_rectangle(x_min, y_min, x_max, y_max, (theta * 2 - 1) * 45)
        x, y = boundary_polygon.exterior.coords.xy
        color = [89, 89, 89]
        for i in range(3):
            color[i] /= 256
        ax.plot(x, y, '-', color=color)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

    boundary_directory = path.replace(model, 'figure_boundary')
    os.makedirs(boundary_directory, exist_ok=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_axis_off()
    save_path_2 = os.path.join(boundary_directory, file_idx + "_boundary.png")
    ax.figure.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close(ax.figure)
