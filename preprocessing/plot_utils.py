import matplotlib.pyplot as plt
import random

def get_random_color(seed):
    random.seed(seed)
    return (random.random(), random.random(), random.random())

def plot_groups_with_rectangles_v7(unit_roads, bounding_boxs, building_polygons):
    # for unit_road_idx, unit_road in enumerate(unit_roads):
    #     x, y = [unit_road[1][0][0], unit_road[1][1][0]], [unit_road[1][0][1], unit_road[1][1][1]]
    #     plt.plot(x, y, color=get_random_color(unit_road[0]))
    #     plt.text((unit_road[1][0][0] + unit_road[1][1][0]) / 2,
    #              (unit_road[1][0][1] + unit_road[1][1][1]) / 2,
    #              unit_road_idx, fontsize=7, ha='center', va='center', color='black')

    for unit_road_idx, unit_road in enumerate(unit_roads):
        x, y = [unit_road[1][0][0], unit_road[1][1][0]], [unit_road[1][0][1], unit_road[1][1][1]]
        plt.plot(x, y, color=get_random_color(unit_road[0]))
        plt.text((unit_road[1][0][0] + unit_road[1][1][0]) / 2,
                 (unit_road[1][0][1] + unit_road[1][1][1]) / 2,
                 unit_road[0], fontsize=7, ha='center', va='center', color='black')

    for bounding_box_idx, bounding_box in enumerate(bounding_boxs):
        x, y = bounding_box[1].exterior.xy
        plt.plot(x, y, color=get_random_color(bounding_box[0]))
        centroid = bounding_box[1].centroid
        plt.text(centroid.x, centroid.y,
                 bounding_box[0], fontsize=7, ha='center', va='center', color='black')

    for building_polygon_idx, building_polygon in enumerate(building_polygons):
        x, y = building_polygon[2].exterior.xy
        plt.plot(x, y, color=get_random_color(building_polygon[0]))
        centroid = building_polygon[2].centroid
        plt.text(centroid.x, centroid.y,
                 building_polygon[1], fontsize=7, ha='center', va='center', color='black')

    plt.show()
