import concurrent.futures
from tqdm import tqdm
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import networkx as nx

def create_random_graph(n, p):
    # n: 노드 수, p: 에지 존재 확률
    G = nx.erdos_renyi_graph(n, p)
    return list(G.edges)

def create_ring_graph(n):
    edges = []
    for i in range(n):
        edges.append((i, i))

    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    edges.append((0, n - 1))
    edges.append((n - 1, 0))

    return edges

def create_line_graph(n):
    edges = []
    for i in range(n):
        edges.append((i, i))

    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    return edges

def create_grid_graph(n1, n2):
    edges = []  # 에지 리스트
    for i in range(n2):  # 세로 크기
        for j in range(n1):  # 가로 크기
            # 현재 노드 인덱스 계산
            node = i * n1 + j
            # 우측 노드와 연결 (열 내)
            if j < n1 - 1:
                edges.append((node, node + 1))
                edges.append((node + 1, node))
            # 하단 노드와 연결 (행 내)
            if i < n2 - 1:
                edges.append((node, node + n1))
                edges.append((node + n1, node))

    return edges


def edges_to_adj_matrix(edges):
    # 가장 큰 노드 번호 찾기
    max_node = max(max(edge) for edge in edges)
    # 노드 개수 (가장 큰 번호 + 1)
    n = max_node + 1

    # 모든 원소가 0인 n x n 행렬 생성
    adj_matrix = np.zeros((n, n), dtype=int)

    # 에지 리스트를 순회하면서 인접 행렬 채우기
    for edge in edges:
        i, j = edge
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1  # 무방향 그래프를 가정

    return adj_matrix

if __name__ == '__main__':
    folder_path = 'C:/Users/Dobby/Downloads/large_scale_datasets/image_condition_train_datasets/'
    gpickle_files = [f for f in os.listdir(folder_path) if f.endswith('.gpickle')]
    for file_path in tqdm(gpickle_files):
        load_path = folder_path + '/' + file_path
        with open(load_path, 'rb') as f:
            graph = pickle.load(f)

        ffile = file_path.replace('.gpickle', '.pkl')
        with open(f'C:/Users/Dobby/Downloads/large_scale_datasets/graph_condition_train_datasets/{ffile}', 'rb') as file:
            buildings = pickle.load(file)

        n_node = graph.number_of_nodes()
        n_building = len(buildings)
        n_chunk = n_node - n_building

        adj_matrix = nx.adjacency_matrix(graph).todense()
        boundary_adj_matrix = adj_matrix[:n_chunk, :n_chunk]
        building_adj_matrix = adj_matrix[n_chunk:, n_chunk:]
        bb_adj_matrix = adj_matrix[n_chunk:, :n_chunk]
        boundary_pos_feature = []

        for node in graph.nodes():
            if node < n_chunk:
                boundary_pos_feature.append(graph.nodes[node]['node_features'][:2])
        boundary_pos_feature = np.array(boundary_pos_feature)

        data = {'boundary_adj_matrix': boundary_adj_matrix,
                'building_adj_matrix': building_adj_matrix,
                'bb_adj_matrix': bb_adj_matrix,
                'boundary_pos_feature': boundary_pos_feature,
                'n_boundary': n_chunk,
                'n_building': n_building}

        output_file_path = 'C:/Users/Dobby/Downloads/large_scale_datasets/graph_generation_datasets/'
        ffile = file_path.replace('.gpickle', '.pickle')
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        with open(f'{output_file_path}/{ffile}', 'wb') as f:
            pickle.dump(data, f)

        # # 그래프 시각화
        # plt.figure(figsize=(8, 6))
        # nx.draw(G_visualized, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
        # plt.title("Random Graph Visualization")
        # plt.show()
