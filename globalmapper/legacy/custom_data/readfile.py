import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
import numpy as np

# gpickle 파일 로드

globalmapper_data_root = "../globalmapper_dataset/processed"
gpickle_data = os.path.join(globalmapper_data_root, "0.gpickle")

globalmapper_raw_data_root  = "../globalmapper_dataset/raw_geo"
globalmapper_raw_data = os.path.join(globalmapper_raw_data_root, "0")

globalmapper_custom_raw_data_root = "../globalmapper_custom_data/raw_geo"
globalmapper_custom_raw_data = os.path.join(globalmapper_custom_raw_data_root, "95.pkl")



G = nx.read_gpickle(gpickle_data)

# with open(globalmapper_raw_data, 'rb') as file:
#     data = pickle.load(file)
#
# with open(globalmapper_custom_raw_data, 'rb') as file:
#     raw_data = pickle.load(file)
#
# print(raw_data)



def mask_to_polygons(mask):
    # 연결된 구성 요소 탐지
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    polygons = []
    for i in range(1, num_labels):  # 0은 배경이므로 제외
        component_mask = (labels == i).astype(np.uint8)

        # 외곽선 탐지
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] >= 3:  # 폴리곤을 형성하기 위해서는 최소 3개의 점이 필요합니다.
                contour = contour.squeeze(axis=1)  # (num_points, 1, 2) -> (num_points, 2)
                polygon = Polygon(contour)
                polygons.append(polygon)

    return polygons


# 그래프 정보 출력
for node, data in G.nodes(data=True):
    print(f"Node: {node}")
    print(f"Data: {data}")
    print(f"Graph: {G.graph}")
    print("------")

# edges = list(G.edges())
# print(edges)

import networkx as nx

# G = nx.Graph()
new_G = nx.Graph(G.edges())

stubby_edge_folder_path = os.path.join("Z:", "iiixr-drive", "Projects", "2023_City_Team", "1_evaluation", "globalmapper")
stubby_edge_file_path = os.path.join(stubby_edge_folder_path, "stubby_edge.gpickle")

nx.write_gpickle(new_G, stubby_edge_file_path)


# for node, data in G.nodes(data=True):
#     print(f"label: {data['old_label']}")
#     print(f"posx: {data['posx']}")
#     print(f"posy: {data['posy']}")
#     print("------")


# # posx와 posy가 모두 0인 노드를 찾아서 그래프에서 제거
# remove_nodes = [node for node in G.nodes() if G.nodes[node]['posx'] == 0 and G.nodes[node]['posy'] == 0]
# G.remove_nodes_from(remove_nodes)
#
# # posx, posy 속성을 기반으로 위치 정보 생성
# pos = {node: (G.nodes[node]['posx'], G.nodes[node]['posy']) for node in G.nodes()}
#
# # 그래프 시각화
# nx.draw(G, pos, with_labels=True, node_size=15, node_color='skyblue', font_size=15)
# plt.show()


# pos = {node: (G.nodes[node]['posx'], G.nodes[node]['posy']) for node in G.nodes()}
#
# # 그래프 시각화
# nx.draw(G, pos, with_labels=True, node_size=15, node_color='skyblue', font_size=15)
# plt.show()
#
# def plot_mask(mask):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(mask, cmap='gray')
#     plt.title("Mask Visualization")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.colorbar()
#     plt.show()
#
# plot_mask(mask)