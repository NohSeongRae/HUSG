import os
import json
import csv

# city_name = 'milan'

city_list = ['portland']

def normalize(value, min_value, range_value):
    return (value - min_value) / range_value

for idx in range(len(city_list)):
    city_name = city_list[idx]

    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'density20_building120_rotate_normalized', 'Buildings')
    files = os.listdir(dir_path)
    filenum = len(files)

    j = 0

    for i in range(1, filenum+1):
        test_boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'density20_building120_rotate_normalized',
                                              'Buildings',
                                        f'{city_name}_buildings{i}.geojson')

        j +=1

        if os.path.exists(test_boundary_filename):

            with open(test_boundary_filename, "r") as f:
                geojson_boundary = json.load(f)

            coordinates = []

            for feature in geojson_boundary['features']:
                coordinates.append(feature['geometry']['coordinates'][0])

            coordinates = coordinates[0]

            # x, y 좌표의 최소, 최대값 찾기
            x_values = [coord[0] for coord in coordinates]
            y_values = [coord[1] for coord in coordinates]

            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            # x, y의 범위 계산
            range_x = max_x - min_x
            range_y = max_y - min_y

            # 가장 큰 범위 선택
            max_range = max(range_x, range_y)

            # 좌표 데이터 정규화
            normalized_coordinates = [[normalize(x, min_x, max_range), normalize(y, min_y, max_range)] for x, y in coordinates]

            csv_folder = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildingcsv')

            if not os.path.exists(csv_folder):
                os.makedirs(csv_folder)

            csv_boundary_filename = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildingcsv',
                                            f'{city_name}_buildings{j}.csv')

            with open(csv_boundary_filename, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(normalized_coordinates[:-1])

