import matplotlib.pyplot as plt
import random
import numpy as np
from shapely.geometry import Point
import os
import seaborn as sns
import matplotlib.pyplot as plt

# def get_random_color(seed):
#     random.seed(seed)
#     return (random.random(), random.random(), random.random())

def get_random_color(seed, palette='pastel', n_colors=30):
    random.seed(seed)
    colors = sns.color_palette(palette, n_colors)
    return colors[random.randint(0, len(colors) - 1)]


def extract_before_underscore(s):
    return s.split('_')[0]

def plot_groups_with_rectangles_v7(block_gdf, buildings_gdf, unit_roads, bounding_boxs, building_polygons, adj_matrix, n_street, street_position_dataset, file_name):
    # fig2, plt = plt.subplots(figsize=(8, 8))
    # block_gdf.boundary.plot(ax=plt, color='blue', label='Rotated Block Boundary')
    # buildings_gdf.plot(ax=plt, color='red', label='Rotated Buildings')

    for unit_road_idx, unit_road in enumerate(unit_roads):
        x, y = [unit_road[1][0][0], unit_road[1][1][0]], [unit_road[1][0][1], unit_road[1][1][1]]
        plt.plot(x, y, color=get_random_color(unit_road[0]))
        plt.text((unit_road[1][0][0] + unit_road[1][1][0]) / 2,
                 (unit_road[1][0][1] + unit_road[1][1][1]) / 2,
                 unit_road_idx, fontsize=7, ha='center', va='center', color='black')

    # for bounding_box_idx, bounding_box in enumerate(bounding_boxs):
    #     x, y = bounding_box[1].exterior.xy
    #     plt.plot(x, y, color=get_random_color(bounding_box[0]))
    #     centroid = bounding_box[1].centroid
    #     plt.text(centroid.x, centroid.y,
    #              bounding_box[0], fontsize=7, ha='center', va='center', color='black')

    for building_polygon_idx, building_polygon in enumerate(building_polygons):
        x, y = building_polygon[2].exterior.xy
        plt.fill(x, y, color=get_random_color(building_polygon[0]), alpha=0.8)
        centroid = building_polygon[2].centroid
        plt.text(centroid.x, centroid.y,
                 str(building_polygon_idx) + ', ' + str(building_polygon[1]),
                 fontsize=7, ha='center', va='center', color='black')

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] == 1 and adj_matrix[j][i] == 1:
                if i < n_street:
                    node_i = Point(np.mean(street_position_dataset[i], axis=0))
                else:
                    node_i = Point(building_polygons[i - n_street][2].centroid)
                if j < n_street:
                    node_j = Point(np.mean(street_position_dataset[j], axis=0))
                else:
                    node_j = Point(building_polygons[j - n_street][2].centroid)

                plt.plot([node_i.x, node_j.x], [node_i.y, node_j.y])

    # if file_name is not None:
    #     city_name = extract_before_underscore(file_name)
    #     folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'preprocessed_image')
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     image_filename = os.path.join(folder_path, f'{file_name}.png')
    #
    #     plt.savefig(image_filename)
    #     plt.clf()

    plt.show()

def plot_bbox(building_bboxs, unit_road_bboxs, unit_road_street_indcies):

    for i in range(len(building_bboxs)):
        x, y = building_bboxs[i].exterior.coords.xy
        plt.plot(x, y)

    for i in range(len(unit_road_bboxs)):
        x, y = unit_road_bboxs[i].exterior.coords.xy
        plt.plot(x, y, color=get_random_color(unit_road_street_indcies[i]))

def plot_graph(node_features, edge_index):
    for edge in edge_index:
        node_a = node_features[edge[0]][:2]
        node_b = node_features[edge[1]][:2]

        plt.plot([node_a[0], node_b[0]], [node_a[1], node_b[1]])

