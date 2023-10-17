import json
import os
import re
import sys
import math

import numpy as np

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

# from etc import filepath as filepath
import matplotlib.pyplot as plt
from shapely.geometry import shape

def calculate_area(circumference):
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2
    return area

def ratio(city_name_list, rangesemantic):
    city_ratio_list = []

    for cityidx in range(len(city_name_list)):
        filelist = []

        import csv

        with open(filepath.removed_filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                number = int(row[0])
                filelist.append(number)

        dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset', 'Boundaries')
        files = os.listdir(dir_path)
        filenum = len(files)

        boundary_length_list = []
        boundary_area_list = []
        circle_area_list = []
        boundary_circle_list = []

        for i in range(1, filenum + 1):
            if i in filelist:
                boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                                 'Boundaries', f'{city_name_list[cityidx]}_boundaries{i}.geojson')

                if os.path.exists(boundary_filename):
                    with open(boundary_filename, "r", encoding='UTF-8') as file:
                        boundary_data = json.load(file)

                for feature in boundary_data['features']:
                    boundary_geometry = shape(feature['geometry'])

                boundary_length = boundary_geometry.length
                circle_area = calculate_area(boundary_length)
                boundary_area = boundary_geometry.area

                boundary_circle_ratio = (boundary_area / circle_area) * 100

                boundary_length_list.append(boundary_length)
                boundary_area_list.append(boundary_area)
                circle_area_list.append(circle_area)

                if boundary_circle_ratio > 90:
                    boundary_circle_list.append(90)
                if boundary_circle_ratio > 80:
                    boundary_circle_list.append(80)
                if boundary_circle_ratio > 70:
                    boundary_circle_list.append(70)
                if boundary_circle_ratio > 60:
                    boundary_circle_list.append(60)
                if boundary_circle_ratio > 50:
                    boundary_circle_list.append(50)
                if boundary_circle_ratio > 40:
                    boundary_circle_list.append(40)
                if boundary_circle_ratio > 30:
                    boundary_circle_list.append(30)
                if boundary_circle_ratio > 20:
                    boundary_circle_list.append(20)
                if boundary_circle_ratio > 10:
                    boundary_circle_list.append(10)
                if boundary_circle_ratio > 0:
                    boundary_circle_list.append(0)

        city_ratio_list.append(boundary_circle_list)

    subplot_height = 8 / 5
    subplot_width = 16 / 3

    if rangesemantic == "_0_15":
        rows = 3
        cols = 5

    else:
        rows = 2
        cols = 5

    # figsize_width = subplot_width * rows
    # figsize_height = subplot_height * cols

    figsize_width = 20
    figsize_height = 8

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, sharex=True,
                            figsize=(figsize_width, figsize_height))

    axs = axs.ravel()

    for idx, axs in enumerate(axs):
        axs.hist(city_ratio_list[idx], rwidth=0.8, color='skyblue')
        axs.set_title(city_name_list[idx])
        # axs.set_axis('off')

    citystat_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'citystatistics', 'pngfiles',
                                 f'ratio{rangesemantic}.png')

    plt.savefig(citystat_path)



if __name__ == "__main__":
    city_name_list = ["atlanta", "barcelona", "budapest", "dallas", "dublin", "firenze", "houston", "lasvegas", "littlerock", "manchester", "milan", "minneapolis",
                      "nottingham", "paris", "philadelphia", "phoenix", "portland", "richmond", "saintpaul", "sanfrancisco", "singapore", "toronto", "vienna",
                      "washington", "zurich"]

    ratio(city_name_list[15:], rangesemantic="_15_25")


