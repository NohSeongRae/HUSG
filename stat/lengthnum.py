import json
import os
import re
import sys
import math
from tqdm import tqdm
import numpy as np

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath
import matplotlib.pyplot as plt
from shapely.geometry import shape


def calculate_area(circumference):
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2
    return area


def lengthnum(city_name_list, filtered_list):
    city_lengthnum_list = []

    filtered_lengthnum_list = []

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

        for i in tqdm(range(1, filenum + 1)):
            if i in filelist:
                if i in filtered_list:
                    boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                                     'Boundaries', f'{city_name_list[cityidx]}_boundaries{i}.geojson')

                    building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                                     'Combined_Buildings', f'{city_name_list[cityidx]}_buildings{i}.geojson')

                    if os.path.exists(building_filename):
                        print("n", boundary_filename)
                        with open(boundary_filename, "r", encoding='UTF-8') as file:
                            boundary_data = json.load(file)
                        print(boundary_data)
                    else:
                        print("nothing")

                    for feature in boundary_data['features']:
                        boundary_geometry = shape(feature['geometry'])

                    boundary_length = boundary_geometry.length

                    # print(boundary_length)

                    if boundary_length >= 0 and boundary_length <= 0.02:
                        filtered_lengthnum_list.append(i)

                # boundary_length_list.append(boundary_length)

        # city_lengthnum_list.append(boundary_length_list)


    """

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
        axs.hist(city_lengthnum_list[idx], bins=40, range=(0.0, 0.04), rwidth=0.8, color='skyblue')
        axs.set_title(city_name_list[idx], fontsize=6)
        # axs.set_axis('off')

    citystat_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'citystatistics', 'pngfiles',
                                 f'lengthnum{rangesemantic}.png')

    plt.savefig(citystat_path)
    
    """

    return filtered_lengthnum_list


if __name__ == "__main__":
    city_name_list = ["atlanta", "barcelona", "budapest", "dallas", "dublin", "firenze", "houston", "lasvegas", "littlerock", "manchester", "milan", "minneapolis",
                      "nottingham", "paris", "philadelphia", "phoenix", "portland", "richmond", "saintpaul", "sanfrancisco", "singapore", "toronto", "vienna",
                      "washington", "zurich"]


    lengthnum(city_name_list[15:], rangesemantic="_15_25")
