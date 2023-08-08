import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import json

current_script_path = os.path.dirname(os.path.abspath(__file__))
husg_directory_path = os.path.dirname(current_script_path)
sys.path.append(husg_directory_path)

from etc import filepath as filepath
from etc.cityname import city_name


def polar_angle(origin, point):
    delta_x = point[0] - origin[0]
    delta_y = point[1] - origin[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle if angle >= 0 else 2 * np.pi + angle

def graph_dataloader(city_name):
    dir_path = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Boundaries')
    files = os.listdir(dir_path)
    num_file = len(files)

    # CSV 파일 경로
    csv_filepath = filepath.graph_filepath

    city_name = 'minneapolis'
    # csv_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset',
    #                              f'{city_name}_graph.csv')

    for i in range(1, num_file + 1):
        boundary_filepath = os.path.join('Z:', 'iiixr-drive', 'Projects', '2023_City_Team', f'{city_name}_dataset', 'Buildings', f'{city_name}_buildings{i}.geojson')

        if os.path.exists(boundary_filepath):
            with open(boundary_filepath, "r", encoding='UTF8') as infile:
                boundary_geojsonfile = json.load(infile)

        boundary_geojsonfile

    # CSV 파일 읽기
    df = pd.read_csv(csv_filepath)

    # 그룹별로 정렬 및 그래프 생성
    graph_list = []
    for group_id, group_df in df.groupby('group'):
        # 그룹 내 좌표 정보 추출
        point_list = group_df[['centroid.x', 'centroid.y']].values

        if len(point_list) == 0:
            continue

        point_list = np.unique(point_list, axis=0)

        print(point_list)

        # 좌표 정렬
        # x축과 양의 방향으로 가장 가까운 점부터 시작
        sorted_indices = np.argsort([polar_angle(np.mean(point_list, axis=0), point) for point in point_list])
        sorted_points = point_list[sorted_indices]

        G = nx.Graph()

        for i, point in enumerate(sorted_points):
            G.add_node(i, pos=point)

        for i in range(len(sorted_points)):
            G.add_edge(i, (i+1) % len(sorted_points))

        graph_list.append(G)

    return graph_list

# 그래프 시각화
def plot_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')

    nx.draw(graph, pos, node_color='blue', node_size=10)
    labels = nx.draw_networkx_labels(graph, pos, font_color='red')
    for node, label in labels.items():
        label.set_text(f"{node}")

    plt.show()

if __name__=='__main__':
    graph_list = graph_dataloader(city_name)

    for graph in graph_list:
        plot_graph(graph)
