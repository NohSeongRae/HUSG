import json
import os
import re
import sys
import math

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath
import matplotlib.pyplot as plt
from shapely.geometry import shape
from matplotlib.ticker import ScalarFormatter


def calculate_area(circumference):
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2
    return area

def lengtharea(city_name_list, rangesemantic):
    city_boundarylength_list = []
    city_boundaryarea_list = []
    city_circlearea_list = []

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

                boundary_length_list.append(boundary_length)
                boundary_area_list.append(boundary_area)
                circle_area_list.append(circle_area)

        sorted_values = sorted(zip(boundary_length_list, boundary_area_list, circle_area_list))
        boundary_length_list, boundary_area_list, circle_area_list = zip(*sorted_values)

        city_boundarylength_list.append(boundary_length_list)
        city_boundaryarea_list.append(boundary_area_list)
        city_circlearea_list.append(circle_area_list)

    subplot_height = 8 / 5
    subplot_width = 16 / 3

    if rangesemantic == "_0_15":
        rows = 3
        cols = 5

    else:
        rows = 2
        cols = 5

    figsize_width = subplot_width * rows
    figsize_height = subplot_height * cols

    # figsize_width = 20
    # figsize_height = 8

    fig, axs = plt.subplots(rows, cols, constrained_layout=True, sharex=True, sharey=True,
                            figsize=(figsize_width, figsize_height))

    axs = axs.ravel()

    # print(city_boundarylength_list)
    # print(city_boundaryarea_list)
    print(circle_area_list)

    for idx, axs in enumerate(axs):
        axs.plot(city_boundarylength_list[idx], city_boundaryarea_list[idx], color='skyblue')
        axs.plot(city_boundarylength_list[idx], city_circlearea_list[idx], color='pink')
        axs.set_title(city_name_list[idx])
        axs.set_xlim([0.0, 0.04])
        axs.set_ylim([0.0, 1e-4])

        axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    citystat_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', 'citystatistics', 'pngfiles',
                                 f'lengtharearatio{rangesemantic}.png')


    plt.rcParams['font.size'] = 6
    plt.tight_layout()

    plt.savefig(citystat_path)


if __name__ == "__main__":
    city_name_list = ["atlanta", "barcelona", "budapest", "dallas", "dublin", "firenze", "houston", "lasvegas", "littlerock", "manchester", "milan", "minneapolis",
                      "nottingham", "paris", "philadelphia", "phoenix", "portland", "richmond", "saintpaul", "sanfrancisco", "singapore", "toronto", "vienna",
                      "washington", "zurich"]


    lengtharea(city_name_list[:15], rangesemantic="_0_15")


