import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np
import pickle
import os

class GraphDataset(Dataset):
    def __init__(self, data_type='train', transform=None, pre_transform=None, condition_type='graph'):
        super(GraphDataset, self).__init__(transform, pre_transform)

        self.condition_type = condition_type
        self.data_type = data_type

        if condition_type == 'graph':
            self.folder_path = '/local_datasets/graph_condition_train_datasets/' + self.data_type
        elif condition_type == 'image':
            self.folder_path = '/local_datasets/image_condition_train_datasets/' + self.data_type
        elif condition_type == 'image_resnet34':
            self.folder_path = '/local_datasets/image_resnet34_condition_train_datasets/' + self.data_type
        file_extension = '.gpickle'

        count = 0
        try:
            for filename in os.listdir(self.folder_path):
                if filename.endswith(file_extension):
                    count += 1
        except:
            self.folder_path = self.folder_path.replace('/data2', '')
            for filename in os.listdir(self.folder_path):
                if filename.endswith(file_extension):
                    count += 1
        self.gpickle_files = [f for f in os.listdir(self.folder_path) if f.endswith('.gpickle')]
        self.gpickle_files.sort()

        # 올바른 파일 목록을 저장할 리스트
        self.valid_files = []

        # 각 .gpickle 파일을 순회하며 검사합니다.
        for file_name in self.gpickle_files:
            file_path = os.path.join(self.folder_path, file_name)
            # 파일을 로드합니다.
            G = nx.read_gpickle(file_path)

            building_masks = torch.tensor(np.array([G.nodes[node]['building_masks'] for node in G.nodes()]),
                                          dtype=torch.long)

            if torch.sum(building_masks) > 1:
                self.valid_files.append(file_name)

        self.data_length = len(self.valid_files)
        print(self.data_length)

    def get(self, idx):
        if self.data_type == 'train' or self.data_type == 'val':
            load_path = self.folder_path + '/' + self.valid_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)

            graph = self.graph

            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)

            if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
                condition = torch.tensor(np.array(graph.graph['condition']), dtype=torch.float32)
            else:
                condition_graph = graph.graph['condition']
                condition_edge_index = nx.to_scipy_sparse_matrix(condition_graph).tocoo()
                condition_edge_index = torch.tensor(np.vstack((condition_edge_index.row, condition_edge_index.col)),
                                                    dtype=torch.long)
                condition_street_feature = torch.tensor(
                    np.array([condition_graph.nodes[node]['chunk_features'] for node in condition_graph.nodes()]),
                    dtype=torch.float32)

                condition = Data(condition_street_feature=condition_street_feature,
                                 edge_index=condition_edge_index,
                                 num_nodes=condition_graph.number_of_nodes())

            grid_graph = self.make_grid_graph(node_features, building_masks)
            node_features = torch.tensor(np.array([grid_graph.nodes[node]['node_features'] for node in grid_graph.nodes()]),
                                         dtype=torch.float32)
            exist_features = torch.tensor(np.array([grid_graph.nodes[node]['exist_features'] for node in grid_graph.nodes()]),
                                          dtype=torch.long)

            edge_index = nx.to_scipy_sparse_matrix(grid_graph).tocoo()
            edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

            data = Data(node_features=node_features, exist_features=exist_features, condition=condition,
                        edge_index=edge_index, num_nodes=grid_graph.number_of_nodes())

            return data
        else:
            load_path = self.folder_path + '/' + self.valid_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)
            graph = self.graph

            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)

            if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
                condition = torch.tensor(np.array(graph.graph['condition']), dtype=torch.float32)
            else:
                condition_graph = graph.graph['condition']
                condition_edge_index = nx.to_scipy_sparse_matrix(condition_graph).tocoo()
                condition_edge_index = torch.tensor(np.vstack((condition_edge_index.row, condition_edge_index.col)),
                                                    dtype=torch.long)
                condition_street_feature = torch.tensor(
                    np.array([condition_graph.nodes[node]['chunk_features'] for node in condition_graph.nodes()]),
                    dtype=torch.float32)

                condition = Data(condition_street_feature=condition_street_feature,
                                 edge_index=condition_edge_index,
                                 num_nodes=condition_graph.number_of_nodes())

            grid_graph = self.make_grid_graph(node_features, building_masks)
            node_features = torch.tensor(np.array([grid_graph.nodes[node]['node_features'] for node in grid_graph.nodes()]),
                                         dtype=torch.float32)
            exist_features = torch.tensor(np.array([grid_graph.nodes[node]['exist_features'] for node in grid_graph.nodes()]),
                                          dtype=torch.long)

            edge_index = nx.to_scipy_sparse_matrix(grid_graph).tocoo()
            edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

            data = Data(node_features=node_features, exist_features=exist_features, condition=condition,
                        edge_index=edge_index, num_nodes=grid_graph.number_of_nodes())

            polygon_path = self.valid_files[idx].replace('.gpickle', '.pkl')
            return (data, polygon_path, self.valid_files[idx])
    def len(self):
        return self.data_length

    def make_edge(self):
        # 그리드의 크기
        rows, cols = 3, 40

        # 각 노드의 상하좌우 인접 노드와의 연결을 나타내는 간선 인덱스 생성
        edge_indices = []

        for row in range(rows):
            for col in range(cols):
                node_index = row * cols + col  # 현재 노드의 인덱스

                neighbors = [
                    (row - 1, col),  # 상
                    (row + 1, col),  # 하
                    (row, col - 1),  # 좌
                    (row, col + 1)  # 우
                ]

                # edge_indices.append([node_index, node_index])
                for n_row, n_col in neighbors:
                    # 인접 노드가 그리드 범위 내에 있는지 확인
                    if 0 <= n_row < rows and 0 <= n_col < cols:
                        neighbor_index = n_row * cols + n_col
                        # 간선 인덱스에 추가 (방향성이 없는 그래프 가정)
                        edge_indices.append([node_index, neighbor_index])
        return edge_indices

    def graph_node(self):
        graph_nodes_list = []
        for i in range(4):
            for j in range(30):
                graph_nodes_list.append((i, j))

        return graph_nodes_list

    def make_grid_graph(self, node_features, building_masks):
        # 그리드 사이즈 + 2
        rows, cols = 5, 42

        # 행과 열에 대한 분할 점 계산
        y_divisions = np.linspace(0, 1, rows)[1: -1]
        x_divisions = np.linspace(0, 1, cols)[1: -1]

        # 모든 교차점의 x, y 좌표를 계산
        x_coords, y_coords = np.meshgrid(x_divisions, y_divisions)

        # 2차원 좌표 배열로 변환
        coordinates = np.dstack([x_coords, y_coords])
        coordinates = np.reshape(coordinates, (-1, 2))

        node_indices = np.zeros(120, dtype=int)

        used_indices = []
        for idx, data in enumerate(zip(node_features[:, 0], node_features[:, 1], building_masks)):
            x_pos, y_pos, mask = data
            if mask == 0:
                continue

            deltas = coordinates - np.array([x_pos, y_pos])
            dist_squared = np.sum(deltas ** 2, axis=1)
            for used_index in used_indices:
                dist_squared[used_index] = np.inf

            # 가장 짧은 거리의 인덱스 찾기
            closest_index = np.argmin(dist_squared)
            used_indices.append(closest_index)

            # plt.scatter(coordinates[closest_index][0], coordinates[closest_index][1], color='r')
            node_indices[closest_index] = idx + 1

        G = nx.Graph()
        G.add_edges_from(self.make_edge())

        for node in G.nodes():
            if node_indices[node] > 0:
                building_index = node_indices[node] - 1
                G.nodes[node]['node_features'] = node_features[building_index]
                G.nodes[node]['building_masks'] = building_masks[building_index]
                G.nodes[node]['exist_features'] = 1
            else:
                G.nodes[node]['node_features'] = np.zeros_like(node_features[0])
                G.nodes[node]['building_masks'] = np.zeros_like(building_masks[0])
                G.nodes[node]['exist_features'] = 0

        return G