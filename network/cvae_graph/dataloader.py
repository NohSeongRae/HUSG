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
                 condition_type='graph'):
        super(GraphDataset, self).__init__(transform, pre_transform)

        self.condition_type = condition_type
        self.data_type = data_type

        if condition_type == 'graph':
            self.folder_path = '../../../../data2/local_datasets/graph_condition_train_datasets/' + self.data_type
        elif condition_type == 'image':
            self.folder_path = '../../../../data2/local_datasets/image_condition_train_datasets/' + self.data_type
        elif condition_type == 'image_resnet34':
            self.folder_path = '../../../../data2/local_datasets/image_resnet34_condition_train_datasets/' + self.data_type
        file_extension = '.gpickle'  # glob 패턴으로 확장자 설정
        self.folder_path = '../../../../local_datasets/global_mapper/global_mapper/'

        city_names = ["atlanta", "dallas", "houston", "lasvegas", "littlerock",
                      "philadelphia"]
        self.gpickle_files = []
        for city in city_names:
            self.gpickle_files += [city + '/' + f for f in os.listdir(self.folder_path + city) if f.endswith('.gpickle')]
        #
        # if data_type == 'train':
        #     self.train_split_path = 'network/cvae_graph/whole_city/train_split.pkl'
        #     with open(self.train_split_path, 'rb') as f:
        #         self.gpickle_files = pickle.load(f)
        # elif data_type == 'val':
        #     self.val_split_path = 'network/cvae_graph/whole_city/val_split.pkl'
        #     with open(self.val_split_path, 'rb') as f:
        #         self.gpickle_files = pickle.load(f)
        # elif data_type == 'test':
        #     self.test_split_path = 'network/cvae_graph/whole_city/test_split.pkl'
        #     with open(self.test_split_path, 'rb') as f:
        #         self.gpickle_files = pickle.load(f)

        self.data_length = len(self.gpickle_files)

    def get(self, idx):
        if self.data_type == 'train' or self.data_type == 'val':
            # load_path = self.folder_path + '/' + str(idx) + '.gpickle'
            load_path = self.folder_path + '/' + self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)

            # 그래프 리스트에서 인덱스에 해당하는 그래프를 선택합니다.
            graph = self.graph

            # 그래프를 PyG 데이터 객체로 변환합니다.
            # 노드 특성과 엣지 인덱스를 추출합니다.
            node_features = torch.tensor(np.array([[graph.nodes[node]['posx'], graph.nodes[node]['posy'],
                                                    graph.nodes[node]['size_x'], graph.nodes[node]['size_y'], 0
                                                    ] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['exist'] for node in graph.nodes()]),
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

            edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
            edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

            # PyG 데이터 객체를 생성합니다.
            data = Data(node_features=node_features,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

            return data
        else:
            # load_path = self.folder_path + '/' + str(idx) + '.gpickle'
            load_path = self.folder_path + '/' + self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)

            # 그래프 리스트에서 인덱스에 해당하는 그래프를 선택합니다.
            graph = self.graph

            # 그래프를 PyG 데이터 객체로 변환합니다.
            # 노드 특성과 엣지 인덱스를 추출합니다.
            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            node_semantics = torch.tensor(np.array([graph.nodes[node]['node_semantics'] for node in graph.nodes()]),
                                          dtype=torch.long)
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

            edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
            edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

            mask = ((building_masks[edge_index[0]] == 1) & (building_masks[edge_index[1]] == 1)).squeeze(-1)
            building_edge_index = edge_index[:, mask]

            mask = ((building_masks[edge_index[0]] == 0) & (building_masks[edge_index[1]] == 0)).squeeze(-1)
            street_edge_index = edge_index[:, mask]

            edge_index = torch.cat([building_edge_index, street_edge_index], dim=1)

            # PyG 데이터 객체를 생성합니다.
            data = Data(node_features=node_features, node_semantics=node_semantics,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

            polygon_path = self.gpickle_files[idx].replace('.gpickle', '.pkl')
            return (data, polygon_path, self.gpickle_files[idx])
    def len(self):

        return self.data_length
