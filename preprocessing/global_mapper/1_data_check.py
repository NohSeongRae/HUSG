import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

directory_path = 'raw_datasets/globalmapper_dataset/original'
files = list_files(directory_path)

x_pos_list = []
y_pos_list = []
for file in tqdm(files):
    graph = nx.read_gpickle(file)

    for node in graph.nodes():
        if graph.nodes[node]['exist'] == 1.0:
            x_pos_list.append(graph.nodes[node]['posx'])
            y_pos_list.append(graph.nodes[node]['posy'])

# x에 대한 히스토그램 그리기
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치
plt.hist(x_pos_list, bins=10, color='blue', alpha=0.7)
plt.title('X Positions Distribution')
plt.xlabel('X position')
plt.ylabel('Frequency')

# y에 대한 히스토그램 그리기
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 위치
plt.hist(y_pos_list, bins=10, color='green', alpha=0.7)
plt.title('Y Positions Distribution')
plt.xlabel('Y position')
plt.ylabel('Frequency')

# 그래프 표시
plt.tight_layout()
plt.show()

x_max = np.max(x_pos_list)
x_min = np.min(x_pos_list)
x_mean = np.mean(x_pos_list)
print(f'X Position - Max: {x_max}, Min: {x_min}, Mean: {x_mean:.2f}')

# y 위치에 대한 최대, 최소, 평균 값 계산 및 출력
y_max = np.max(y_pos_list)
y_min = np.min(y_pos_list)
y_mean = np.mean(y_pos_list)
print(f'Y Position - Max: {y_max}, Min: {y_min}, Mean: {y_mean:.2f}')