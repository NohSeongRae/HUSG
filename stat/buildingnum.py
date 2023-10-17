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

from etc import variables as variables
# from etc import filepath as filepath
import matplotlib.pyplot as plt
from shapely.geometry import shape

# from etc.cityname import city_name

def calculate_area(circumference):
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2
    return area

def buildingnum(city_name_list, upperlimit, lowerlimit):
    city_building_num_list = []
    filtered_buildingnum_list = []

    for cityidx in range(len(city_name_list)):

        """
        filelist = []

        import csv

        removed_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset', f'{city_name_list[cityidx]}_removed_filenum.csv')

        with open(removed_filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                number = int(row[0])
                filelist.append(number)
                
        """


        dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset', 'Boundaries')
        files = os.listdir(dir_path)
        filenum = len(files)

        building_num_list = []

        for i in tqdm(range(1, filenum + 1)):
            boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                             'Boundaries', f'{city_name_list[cityidx]}_boundaries{i}.geojson')
            building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name_list[cityidx]}_dataset',
                                             'Combined_Buildings', f'{city_name_list[cityidx]}_buildings{i}.geojson')

            if os.path.exists(boundary_filename):
                with open(boundary_filename, "r", encoding='UTF-8') as file:
                    boundary_data = json.load(file)

            if os.path.exists(building_filename):
                with open(building_filename, "r", encoding='UTF-8') as file:
                    building_data = json.load(file)

                building_geometry = []

                for feature in building_data['features']:
                    building_geometry.append(shape(feature['geometry']))

                # print(building_geometry)

                building_num = len(building_geometry)

                if building_num >= lowerlimit and building_num <= upperlimit:
                    filtered_buildingnum_list.append(i)



                    #if building_num != 0:
                    #    building_num_list.append(building_num)

        # city_building_num_list.append(building_num_list)

    # print(city_building_num_list)

    """

    if rangesemantic == "_0_15":
        rows = 3
        cols = 5

        subplot_height = 8 / 5
        subplot_width = 16 / 3

    else:
        rows = 2
        cols = 5

        subplot_height = 8 / 5
        subplot_width = 16 / 2

    figsize_width = subplot_width * rows
    figsize_height = subplot_height * cols

    # figsize_width = 20
    # figsize_height = 8

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, sharex=True, figsize=(figsize_width, figsize_height))

    axs = axs.ravel()

    for idx, axs in enumerate(axs):
        axs.hist(city_building_num_list[idx], bins=100, range=(0, 100), rwidth=0.8, color='skyblue')
        axs.set_title(city_name_list[idx])

    citystat_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'citystatistics', 'pngfiles', f'buildingnum{rangesemantic}.png')

    plt.savefig(citystat_path)
    
    """
    """
    with open('filtered_buildingnum_list.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(filtered_buildingnum_list)
    """

    return filtered_buildingnum_list


