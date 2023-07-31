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

    for i in range(len(category_list)):
        # building 이 아닌 경우
        if i != len(category_list)-1:
            counts, bins, patches = plt.hist(category_list[i], bins=range(min(category_list[i]), max(category_list[i]) + 2), align='left', rwidth=0.8, color='skyblue')

            plt.title(category_semantic[i])
            plt.xlabel('number')
            plt.ylabel('count')

            plt.xticks(range(min(category_list[i]), max(category_list[i]) + 1))

            for count, bin, patch in zip(counts, bins, patches):
                plt.text(bin, count, str(int(count)), fontsize=10, ha='left')

            plt.savefig(category_filepath[i])
            plt.clf()
        # building
        else:
            list_range = range(min(category_list[i]), max(category_list[i]) + 2)
            thirds = len(list_range) // 25
            first_hist_ylim = None
            for j in range(25):
                start = list_range.start + j * thirds
                end = list_range.start + (j + 1) * thirds if j < 24 else list_range.stop
                counts, bins = np.histogram(category_list[i], bins=range(start, end))  # 빈도 계산
                all_counts.extend(counts)  # 계산된 빈도수를 all_counts에 추가

                valid_bins = bins[:-1][counts >= 2]  # 빈도가 2 이상인 bin 선택
                valid_counts = counts[counts >= 2]  # 빈도가 2 이상인 빈도 선택
                if len(valid_bins) > 0:  # 빈도가 2 이상인 데이터가 있는 경우에만 그림을 그립니다.
                    plt.bar(valid_bins, valid_counts, align='edge', width=np.diff(bins)[0] * 0.9,
                            color='skyblue')  # 필터링된 데이터로 히스토그램 그리기, 막대 색깔을 skyblue로 설정
                    plt.title(f"{category_semantic[i]} ({j + 1})")
                    plt.xlabel('number')
                    plt.ylabel('count')
                    plt.xticks(range(start, end), fontsize=7)  # x축 라벨의 글꼴 크기를 8로 설정
                    for bin, count in zip(valid_bins, valid_counts):  # 필터링된 데이터로 텍스트 추가
                        plt.text(bin, count, str(int(count)), fontsize=8, ha='left')  # 텍스트 크기를 8로 설정

                    if first_hist_ylim is None:
                        first_hist_ylim = plt.ylim()  # get the y limit of the first histogram
                    else:
                        plt.ylim(first_hist_ylim)  # set the y limit to match the first histogram

                    plt.savefig(f"{category_filepath[i]}_{j + 1}.png")
                    plt.clf()

            plt.hist(all_counts, bins=range(min(all_counts), max(all_counts) + 2), color='skyblue', rwidth=0.8)
            counts, bins, patches = plt.hist(all_counts, bins=range(min(all_counts), max(all_counts) + 2),
                                             color='skyblue')
            # Filter the data
            filtered_counts = [count for bin, count in zip(bins[:-1], counts) if bin in [0, 1, 2]]

            plt.bar([0, 1, 2], filtered_counts, color='skyblue')
            plt.title(f"Histogram of counts for {category_semantic[i]}")
            plt.xlabel('count')
            plt.ylabel('frequency')

            # Set y-axis range to match maximum count
            plt.xlim(0, 2)
            plt.ylim(0, max(filtered_counts) if filtered_counts else 0)

            # Set x-axis ticks
            plt.xticks([0, 1, 2])

            # Add frequency text on each bar
            for x, count in zip([0, 1, 2], filtered_counts):
                if count > 0:  # print frequency text only when the count is not 0
                    plt.text(x, count, str(int(count)), fontsize=8, ha='center', va='bottom')

            plt.savefig(f"{category_filepath[i]}_sub.png")
            plt.clf()