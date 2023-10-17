import matplotlib.pyplot as plt
import random
import numpy as np
from shapely.geometry import Point
import os
import seaborn as sns

# def get_random_color(seed):
#     random.seed(seed)
#     return (random.random(), random.random(), random.random())

def get_random_color(seed, palette='pastel', n_colors=30):
    random.seed(seed)
    colors = sns.color_palette(palette, n_colors)
    return colors[random.randint(0, len(colors) - 1)]


def extract_before_underscore(s):
    return s.split('_')[0]

def plot_groups_with_rectangles_v7(unit_roads, bounding_boxs, building_polygons, adj_matrix, n_building, street_position_dataset, file_name):
    for unit_road_idx, unit_road in enumerate(unit_roads):
        x, y = [unit_road[1][0][0], unit_road[1][1][0]], [unit_road[1][0][1], unit_road[1][1][1]]
        plt.plot(x, y, color=get_random_color(unit_road[0]))
        plt.text((unit_road[1][0][0] + unit_road[1][1][0]) / 2,
                 (unit_road[1][0][1] + unit_road[1][1][1]) / 2,
                 unit_road_idx, fontsize=7, ha='center', va='center', color='black')

    for bounding_box_idx, bounding_box in enumerate(bounding_boxs):
        x, y = bounding_box[1].exterior.xy
        plt.plot(x, y, color=get_random_color(bounding_box[0]))
        centroid = bounding_box[1].centroid
        plt.text(centroid.x, centroid.y,
                 bounding_box[0], fontsize=7, ha='center', va='center', color='black')

    for building_polygon_idx, building_polygon in enumerate(building_polygons):
        x, y = building_polygon[2].exterior.xy
        plt.fill(x, y, color=get_random_color(building_polygon[0]), alpha=0.8)
        centroid = building_polygon[2].centroid
        plt.text(centroid.x, centroid.y,
                 str(building_polygon_idx) + ', ' + str(building_polygon[1]),
                 fontsize=7, ha='center', va='center', color='black')

    # for i in range(len(adj_matrix)):
    #     for j in range(len(adj_matrix)):
    #         if adj_matrix[i][j] == 1:
    #             if i < n_building:
    #                 node_i = Point(building_polygons[i][2].centroid)
    #             else:
    #                 node_i = Point(np.mean(street_position_dataset[i - n_building + 1], axis=0))
    #             if j < n_building:
    #                 node_j = Point(building_polygons[j][2].centroid)
    #             else:
    #                 node_j = Point(np.mean(street_position_dataset[j - n_building + 1], axis=0))
    #
    #             plt.plot([node_i.x, node_j.x], [node_i.y, node_j.y])

    city_name = extract_before_underscore(file_name)
    folder_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'preprocessed_image')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image_filename = os.path.join(folder_path, f'{file_name}.png')

    # plt.savefig(image_filename)
    # plt.clf()

    plt.show()