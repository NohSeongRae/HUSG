import json
import os
import re
import sys
import math
import csv
from tqdm import tqdm
import numpy as np

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

# from etc import filepath as filepath
import matplotlib.pyplot as plt
from shapely.geometry import shape
from etc.cityname import city_name


def calculate_area(circumference):
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2
    return area


def boundarysize(city_name_list, upperlimit, lowerlimit):
    city_boundary_area_list = []
    filtered_boundary_list = []

    for cityidx in range(len(city_name_list)):
        filelist = []

        """

        removed_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                        f'{city_name_list[cityidx]}_removed_filenum.csv')

        with open(removed_filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                number = int(row[0])
                filelist.append(number)
                
        """

        # city_name = city_name.capitalize()

        dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset', 'Boundaries')
        files = os.listdir(dir_path)
        filenum = len(files)

        boundary_length_list = []
        boundary_area_list = []
        circle_area_list = []
        boundary_circle_list = []


        for i in tqdm(range(1, filenum + 1)):
            # if i in filelist:
            boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                             'Boundaries', f'{city_name_list[cityidx]}_boundaries{i}.geojson')

            building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team',
                                             f'{city_name_list[cityidx]}_dataset',
                                             'Combined_Buildings',
                                             f'{city_name_list[cityidx]}_buildings{i}.geojson')

            if os.path.exists(building_filename):
                with open(boundary_filename, "r", encoding='UTF-8') as file:
                    boundary_data = json.load(file)

                for feature in boundary_data['features']:
                    boundary_geometry = shape(feature['geometry'])

                boundary_area = boundary_geometry.area

                # print(boundary_area)

                if boundary_area >= lowerlimit and boundary_area <= upperlimit:
                    filtered_boundary_list.append(i)

                """boundary size"""

                # boundary_area_list.append(boundary_area)

        # city_boundary_area_list.append(boundary_area_list)

    """histogram (5x3)"""

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
        axs.hist(city_boundary_area_list[idx], bins=100, range=(0.0, 0.000006), rwidth=0.8, color='skyblue')
        axs.set_title(city_name_list[idx])
        # axs.set_axis('off')


    citystat_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'citystatistics', 'pngfiles',
                                 f'boundarysize{rangesemantic}.png')

    plt.savefig(citystat_path)
    
    """

    """

    with open('filtered_boundary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(filtered_boundary_list)
        
    """

    return filtered_boundary_list


if __name__ == "__main__":
    # csv complete

    city_name_list = filepath.city_names

    print(city_name_list)

    rangesemantic = "_15_25"

    boundarysize(city_name_list[15:], rangesemantic)