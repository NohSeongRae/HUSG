import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np
import pickle
import os


class GraphDataset(Dataset):
    """
    Dataset class for boundary data.
    """

    def __init__(self, data_type='train', transform=None, pre_transform=None,
                 condition_type='graph', chunk_graph=True):
        super(GraphDataset, self).__init__(transform, pre_transform)

        self.condition_type = condition_type
        self.chunk_graph = chunk_graph
        self.data_type = data_type

        if condition_type == 'graph':
            self.folder_path = '../../../..//local_datasets/graph_condition_train_datasets/' + self.data_type
        else:
            self.folder_path = '../../../..//local_datasets/image_condition_train_datasets/' + self.data_type
        file_extension = '.gpickle'  # glob 패턴으로 확장자 설정

        count = 0
        for filename in os.listdir(self.folder_path):
            if filename.endswith(file_extension):
                count += 1
        self.data_length = count

    def get(self, idx):
        load_path = self.folder_path + '/' + str(idx) + '.gpickle'
        with open(load_path, 'rb') as f:
            self.graph = pickle.load(f)

        if self.chunk_graph:
            # 그래프 리스트에서 인덱스에 해당하는 그래프를 선택합니다.
            graph = self.graph

            # 그래프를 PyG 데이터 객체로 변환합니다.
            # 노드 특성과 엣지 인덱스를 추출합니다.
            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            node_semantics = torch.tensor(np.array([graph.nodes[node]['node_semantics'] for node in graph.nodes()]),
                                          dtype=torch.long)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.float32)

            if self.condition_type == 'image':
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

            edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
            edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)
            # PyG 데이터 객체를 생성합니다.
            data = Data(node_features=node_features, node_semantics=node_semantics,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

        else:
            # 그래프 리스트에서 인덱스에 해당하는 그래프를 선택합니다.
            graph = self.graph

            # 그래프를 PyG 데이터 객체로 변환합니다.
            # 노드 특성과 엣지 인덱스를 추출합니다.
            street_feature = torch.tensor(np.array([graph.nodes[node]['street_feature'] for node in graph.nodes()]),
                                          dtype=torch.float)
            building_feature = torch.tensor(np.array([graph.nodes[node]['building_feature'] for node in graph.nodes()]),
                                            dtype=torch.float)
            street_masks = torch.tensor(np.array([graph.nodes[node]['street_masks'] for node in graph.nodes()]),
                                        dtype=torch.float)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.float)

            if self.condition_type == 'image':
                condition = torch.tensor(np.array(graph.graph['condition']), dtype=torch.float)
            else:
                condition_graph = graph.graph['condition']
                condition_edge_index = nx.to_scipy_sparse_matrix(condition_graph).tocoo()
                condition_edge_index = torch.tensor(np.vstack((condition_edge_index.row, condition_edge_index.col)),
                                                    dtype=torch.long)
                condition_street_feature = torch.tensor(
                    np.array([condition_graph.nodes[node]['street_feature'] for node in condition_graph.nodes()]),
                    dtype=torch.float)

                condition = Data(condition_street_feature=condition_street_feature,
                                 edge_index=condition_edge_index,
                                 num_nodes=condition_graph.number_of_nodes())

            edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
            edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

            # PyG 데이터 객체를 생성합니다.
            data = Data(street_feature=street_feature, building_feature=building_feature, street_mask=street_masks,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

        return data

    def len(self):

        return self.data_length
