import json
import os
import sys
import math
import numpy as np
from collections import Counter

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import variables as variables
from etc import filepath as filepath
import matplotlib.pyplot as plt
from shapely.geometry import shape

from etc.cityname import city_name

def remove_duplicate_coordinates(features):
    seen_coordinates = set()
    new_features = []

    for feature in features:
        coordinates = feature['geometry']['coordinates']

        # Convert the list of coordinates to a string
        string_coordinates = json.dumps(coordinates)

        if string_coordinates not in seen_coordinates:
            seen_coordinates.add(string_coordinates)
            new_features.append(feature)

    return new_features

def calculate_area(circumference):
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2
    return area


def block_category(city_name):
    filelist = []

    import csv

    with open(filepath.removed_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            number = int(row[0])
            filelist.append(number)

    # print("list : ", filelist)
    city_name = city_name.capitalize()

    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    filenum = len(files)

    result_list = []

    commercial_list = []
    education_list = []
    emergency_list = []
    financial_list = []
    government_list = []
    healthcare_list = []
    public_list = []
    sport_list = []
    building_list = []

    for i in range(1, filenum + 1):
        if i in filelist:
            boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                             'Boundaries', f'{city_name}_boundaries{i}.geojson')

            building_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
                                             'Combined_Buildings', f'{city_name}_buildings{i}.geojson')

            if os.path.exists(boundary_filename):
                with open(boundary_filename, "r", encoding='UTF-8') as file:
                    boundary_data = json.load(file)

            for feature in boundary_data['features']:
                boundary_geometry = shape(feature['geometry'])

            boundary_length = boundary_geometry.length
            circle_area = calculate_area(boundary_length)

            if os.path.exists(building_filename):
                with open(building_filename, "r", encoding='UTF-8') as file:
                    building_data = json.load(file)

                features = building_data['features']
                features = remove_duplicate_coordinates(features)

                if not features:
                    continue
                else:
                    commercial = 0
                    education = 0
                    emergency = 0
                    financial = 0
                    government = 0
                    healthcare = 0
                    public = 0
                    sport = 0
                    building = 0

                    for feature in features:

                        key = feature['properties']['key']

                        if key in variables.commercial:
                            commercial += 1
                        elif key in variables.education:
                            education += 1
                        elif key in variables.emergency:
                            emergency += 1
                        elif key in variables.financial:
                            financial += 1
                        elif key in variables.government:
                            government += 1
                        elif key in variables.healthcare:
                            healthcare += 1
                        elif key in variables.public:
                            public += 1
                        elif key in variables.sport:
                            sport += 1
                        elif key in variables.building:
                            building += 1

                    commercial_list.append(commercial)
                    education_list.append(education)
                    emergency_list.append(emergency)
                    financial_list.append(financial)
                    government_list.append(government)
                    healthcare_list.append(healthcare)
                    public_list.append(public)
                    sport_list.append(sport)
                    building_list.append(building)

    if not os.path.exists(filepath.blocksemantic):
        os.makedirs(filepath.blocksemantic)

    category_list = [commercial_list, education_list, emergency_list, financial_list, government_list, healthcare_list, public_list, sport_list, building_list]
    category_semantic = ["commercial", "education", "emergency", "financial", "government", "healthcare", "public", "sport", "building"]
    category_filepath = [filepath.commercial_filepath, filepath.education_filepath, filepath.emergency_filepath, filepath.financial_filepath,
                         filepath.government_filepath, filepath.healthcare_filepath, filepath.public_filepath, filepath.sport_filepath, filepath.building_filepath]

    all_counts = []
    max_y = None

    for i in range(len(category_list)):
        # For all categories other than "building"
        if i != len(category_list) - 1:
            counts, bins, patches = plt.hist(category_list[i],
                                             bins=range(min(category_list[i]), max(category_list[i]) + 2), align='left',
                                             rwidth=0.8, color='skyblue')

            plt.title(category_semantic[i])
            plt.xlabel('number')
            plt.ylabel('count')

            plt.xticks(range(min(category_list[i]), max(category_list[i]) + 1))

            for count, bin, patch in zip(counts, bins, patches):
                plt.text(bin, count, str(int(count)), fontsize=10, ha='left')

            plt.savefig(category_filepath[i])
            plt.clf()
        # For "building"
        else:
            list_min = min(category_list[i])
            list_max = max(category_list[i])

            bins = np.linspace(list_min, list_max + 1, num=31)

            for j in range(len(bins) - 1):
                bin_start = bins[j]
                bin_end = bins[j + 1]
                in_this_bin = [x for x in category_list[i] if bin_start <= x < bin_end]

                if len(in_this_bin) > 1:
                    counts, _, _ = plt.hist(in_this_bin, bins=range(int(bin_start), int(bin_end) + 2),
                                            color='skyblue', align='mid', rwidth=0.8)

                    if max_y is None:
                        max_y = max(counts)

                    plt.title(f"{category_semantic[i]}_{j}")
                    plt.xlabel('number')
                    plt.ylabel('count')
                    plt.xticks(range(int(bin_start), int(bin_end) + 2), fontsize=7)

                    for num in in_this_bin:
                        plt.text(num, in_this_bin.count(num), str(int(in_this_bin.count(num))), fontsize=8, ha='left')

                    plt.ylim([0, max_y])
                    plt.xlim([bin_start, bin_end])

                    plt.savefig(f"{category_filepath[i]}_{j}.png")
                    plt.clf()