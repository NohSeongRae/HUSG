import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np
import pickle

class GraphDataset(Dataset):
    """
    Dataset class for boundary data.
    """
    def __init__(self, data_type='train', transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(transform, pre_transform)

        load_path = './network/cvae_graph/' + data_type + '_datasets.gpickle'
        with open(load_path, 'rb') as f:
            self.graphs = pickle.load(f)

    def get(self, idx):
        # 그래프 리스트에서 인덱스에 해당하는 그래프를 선택합니다.
        graph = self.graphs[idx]

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
        graph_names = graph.name

        edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
        edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

        # PyG 데이터 객체를 생성합니다.
        data = Data(street_feature=street_feature, building_feature=building_feature, street_mask=street_masks,
                    building_mask=building_masks, edge_index=edge_index, num_nodes=graph.number_of_nodes())

        return [data, graph_names]

    def len(self):
        return len(self.graphs)