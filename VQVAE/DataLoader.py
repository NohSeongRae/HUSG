import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
import numpy as np
import networkx as nx
import pickle
import os

class GraphDataset(Dataset):
    def __init__(self, data_type='train', transform=None, pre_transform=None, condition_type='graph'):
        super(GraphDataset, self).__init__(transform, pre_transform)

        self.base_dir=os.path.dirname(os.path.realpath(__file__)) #file's directory
        self.condition_type=condition_type
        self.data_type=data_type


        if condition_type=='graph':
            self.folder_path=os.path.join(self.base_dir, 'data2', 'local_datasets', 'graph_condition_train_datasets', self.data_type)
        if condition_type=='image':
            pass #시간있으면 나중에 이미지를 주는 방식으로도 실험해보기
        self.file_extension='.gpickle'

        total_data_num=0

        try:
            for filename in os.listdir(self.folder_path):
                if filename.endswith(self.file_extension):
                    total_data_num+=1

        except:
            print("No data in 'data2' directory. Try to find data in local_datasets")
            self.folder_path = self.folder_path.replace('/data2', '')
            for filename in os.listdir(self.folder_path):
                if filename.endswith(self.file_extension):
                    total_data_num += 1

        self.gpickle_files=[f for f in os.listdir(self.folder_path) if f.endswith(self.file_extension)]
        self.gpickle_files.sort()
        self.data_length=len(self.gpickle_files)
        print(f'Data length: {self.data_length}')

        #Debug
        if self.data_length != total_data_num:
            print(f"Data length: {self.data_length} not matches total_data_num: {total_data_num}")

    def get(self, idx):
        if self.data_type=='train' or self.data_type=='val':
            #Debug
            # print(f'Mode: {self.data_type}')
            load_path=self.folder_path+'/'+self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                graph=pickle.load(f)

            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)
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

            data = Data(node_features=node_features,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

            return data

        else: # Inference
            load_path = self.folder_path + '/' + self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                graph = pickle.load(f)

            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)
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

            data = Data(node_features=node_features,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

            polygon_path = self.gpickle_files[idx].replace('.gpickle', '.pkl')
            return (data, polygon_path, self.gpickle_files[idx])

    def len(self):
        return self.data_length