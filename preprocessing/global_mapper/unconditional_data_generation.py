import concurrent.futures
from tqdm import tqdm
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

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
    end_index = 208622 + 1
    data_type = 'test'
    indices = [1004, 1012, 1003, 109, 105, 1063, 1065, 1067, 1071, 1072,
               1078, 1111, 10002, 1109, 10018, 10013, 10073, 10077, 10080, 10073,
               10196, 10207, 10211, 10212, 10273, 10270, 10291, 10292, 10297, 10301,
               10332, 10320, 10399, 10441, 10483, 10698, 10765, 10792, 10835, 10843, 10841]

    graphs = [create_line_graph(6), create_line_graph(10), create_line_graph(16),
              create_ring_graph(6), create_ring_graph(10), create_ring_graph(16),
              create_grid_graph(2, 6), create_grid_graph(3, 4), create_grid_graph(4, 5),
              create_random_graph(5, 0.5), create_random_graph(10, 0.4), create_random_graph(15, 0.3)]

    graph_types = ['line_small', 'line_middle', 'line_large',
                  'ring_small', 'ring_middle', 'ring_large',
                  'grid_small', 'grid_middle', 'grid_large',
                  'random_small', 'random_middle', 'random_large']

    for idx in indices:
        for building_edge, graph_type in zip(graphs, graph_types):
            # 에지 리스트를 사용하여 NetworkX 그래프 객체 생성
            G_visualized = nx.Graph()
            G_visualized.add_edges_from(building_edge)
            pos = nx.spring_layout(G_visualized)

            node_angles = {node: np.arctan2(pos[node][1], pos[node][0]) for node in pos}
            sorted_nodes = sorted(node_angles, key=lambda node: node_angles[node], reverse=True)

            adj_matrix_original = nx.to_numpy_array(G_visualized)

            node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
            new_indices = [node_mapping[node] for node in G_visualized.nodes()]
            if 'grid' in graph_type or 'random' in graph_type:
                building_adj_matrix = adj_matrix_original[np.ix_(new_indices, new_indices)]
            else:
                building_adj_matrix = adj_matrix_original

            with open(f'datasets/graph_condition_train_datasets/{data_type}/{str(idx)}.pkl', 'rb') as file:
                buildings = pickle.load(file)

            graph = nx.read_gpickle(f'datasets/graph_condition_train_datasets/{data_type}/{str(idx)}.gpickle')

            n_node = graph.number_of_nodes()
            n_building = len(buildings)
            n_chunk = n_node - n_building

            adj_matrix = nx.adjacency_matrix(graph).todense()
            boundary_adj_matrix = adj_matrix[:n_chunk, :n_chunk]
            bb_adj_matrix = np.zeros_like(adj_matrix[n_chunk:, :n_chunk])
            boundary_pos_feature = []

            for node in graph.nodes():
                if node < n_chunk:
                    boundary_pos_feature.append(graph.nodes[node]['node_features'][:2])
            boundary_pos_feature = np.array(boundary_pos_feature)

            n_building = max(max(edge) for edge in building_edge) + 1

            data = {'boundary_adj_matrix': boundary_adj_matrix,
                    'building_adj_matrix': building_adj_matrix,
                    'bb_adj_matrix': bb_adj_matrix,
                    'boundary_pos_feature': boundary_pos_feature,
                    'n_boundary': n_chunk,
                    'n_building': n_building}

            output_file_path = f'random_graph_generation_datasets/{data_type}/'
            with open(f'{output_file_path}/{graph_type}_{idx}.pickle', 'wb') as f:
                pickle.dump(data, f)

            # # 그래프 시각화
            # plt.figure(figsize=(8, 6))
            # nx.draw(G_visualized, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
            # plt.title("Random Graph Visualization")
            # plt.show()
