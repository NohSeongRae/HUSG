import numpy as np
import pandas as pd
import os

city_list = ['portland']

pointnum = 64

def calculate_edge_length(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def generate_points_along_edge(start, end, num_points):
    return np.column_stack((np.linspace(start[0], end[0], num_points + 1)[:-1],
                            np.linspace(start[1], end[1], num_points + 1)[:-1]))


def distribute_points_on_polygon(vertices, total_points=pointnum):
    # Calculate perimeter
    perimeter = sum(calculate_edge_length(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))

    # Calculate how many points to distribute on each edge
    points_distribution = [
        int(np.round(calculate_edge_length(vertices[i], vertices[(i + 1) % len(vertices)]) / perimeter * total_points))
        for i in range(len(vertices) - 1)]  # 마지막 변을 제외하고 계산

    # 최종 점의 수를 596개로 조절하기 위해 마지막 변에 필요한 점의 수 계산
    last_edge_points = total_points - sum(points_distribution)
    points_distribution.append(max(last_edge_points, 1))  # 마지막 변에 최소 1개의 점이 할당되도록 수정

    # Generate points for each edge
    all_points = []
    for i, num_points_edge in enumerate(points_distribution):
        all_points.append(generate_points_along_edge(vertices[i], vertices[(i + 1) % len(vertices)], num_points_edge))

    # Concatenate the points
    points = np.concatenate(all_points)[:total_points]

    return points

for idx in range(len(city_list)):
    city_name = city_list[idx]

    # Specify the directory path
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildingcsv')

    output_dir = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', f'Buildingcsv_{pointnum}')

    # Create the 'distribute' directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all CSV files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(dir_path, filename)

            # Read polygon vertices from CSV file
            df = pd.read_csv(filepath, header=None)
            vertices = df.values.tolist()

            points = distribute_points_on_polygon(vertices)
            output_df = pd.DataFrame(points)

            # Create a new filename for the output CSV
            output_filename = filename.split('.csv')[0] + '.csv'
            output_filepath = os.path.join(output_dir, output_filename)
            output_df.to_csv(output_filepath, index=False, header=False)