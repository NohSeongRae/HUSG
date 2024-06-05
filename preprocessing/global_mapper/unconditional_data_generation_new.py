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

def create_grid_graph_file(path):
    with open(f'C:/Users/Dobby/Downloads/graph_condition_city_datasets/ours_city_datasets/graph_condition_train_datasets/test/{path}.pkl', 'rb') as file:
        buildings = pickle.load(file)
    n_building = len(buildings)

    path_ = f'C:/Users/Dobby/Downloads/graph_condition_city_datasets/ours_city_datasets/graph_condition_train_datasets/test/{path}.gpickle'
    graph = nx.read_gpickle(path_)
    n_boundary = len(graph.nodes) - n_building

    if path == '10921':
        edges_to_remove = [(78, 79), (79, 78),
                           (77, 86), (86, 77),
                           (79, 85), (85, 79),
                           (80, 86), (86, 80),
                           (80, 84), (84, 80),
                           (82, 84), (84, 82)]
        graph.remove_edges_from(edges_to_remove)

    nodes_to_remove = [node for node in graph.nodes() if node < n_boundary]
    graph.remove_nodes_from(nodes_to_remove)

    # pos = nx.spring_layout(graph)
    #
    # plt.figure(figsize=(8, 6))
    # nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
    # plt.title("Random Graph Visualization")
    # plt.show()

    adj_matrix = nx.adjacency_matrix(graph).todense()
    edges = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                edges.append((i, j))
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

def create_grid_graph_ring_based():
    edges = create_ring_graph(8)
    edges.append((0, 8))
    edges.append((8, 0))
    edges.append((2, 8))
    edges.append((8, 2))
    edges.append((4, 8))
    edges.append((8, 4))
    edges.append((6, 8))
    edges.append((8, 6))

    return edges

if __name__ == '__main__':
    data_type = 'test'
    indicis = [11028, 10921, 10507, 1041]
    graphs = [create_ring_graph(6),
              create_grid_graph_file('10921'), create_grid_graph_ring_based(),
              create_random_graph(5, 0.3)]

    for idx, building_edge in enumerate(graphs):
        # 에지 리스트를 사용하여 NetworkX 그래프 객체 생성
        G_visualized = nx.Graph()
        G_visualized.add_edges_from(building_edge)
        pos = nx.spring_layout(G_visualized)

        node_angles = {node: np.arctan2(pos[node][1], pos[node][0]) for node in pos}
        sorted_nodes = sorted(node_angles, key=lambda node: node_angles[node], reverse=True)

        adj_matrix_original = nx.to_numpy_array(G_visualized)

        node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
        new_indices = [node_mapping[node] for node in G_visualized.nodes()]

        if idx >= len(graphs) / 2 + 2:
            building_adj_matrix = adj_matrix_original[np.ix_(new_indices, new_indices)]
        else:
            building_adj_matrix = adj_matrix_original


        with open(f'C:/Users/Dobby/Downloads/graph_condition_city_datasets/ours_city_datasets/graph_condition_train_datasets/{data_type}/{str(indicis[idx])}.pkl', 'rb') as file:
            buildings = pickle.load(file)

        graph = nx.read_gpickle(f'C:/Users/Dobby/Downloads/graph_condition_city_datasets/ours_city_datasets/graph_condition_train_datasets/{data_type}/{str(indicis[idx])}.gpickle')

        n_node = graph.number_of_nodes()
        n_building = len(buildings)
        n_chunk = n_node - n_building

        n_building = max(max(edge) for edge in building_edge) + 1

        adj_matrix = nx.adjacency_matrix(graph).todense()
        boundary_adj_matrix = adj_matrix[:n_chunk, :n_chunk]
        bb_adj_matrix = np.zeros((n_building, n_chunk))
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

        output_file_path = f'random_graph_generation_datasets/'
        with open(f'{output_file_path}/{indicis[idx]}.pickle', 'wb') as f:
            pickle.dump(data, f)

        # 그래프 시각화
        plt.figure(figsize=(8, 6))
        nx.draw(G_visualized, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
        plt.title("Random Graph Visualization")
        plt.show()
