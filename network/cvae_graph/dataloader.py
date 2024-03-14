import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np
import pickle
import os

class GraphDataset(Dataset):
    def __init__(self, data_type='train', transform=None, pre_transform=None, condition_type='graph'):
        """
        Initializes the GraphDataset.

        Parameters:
        - data_type (str): Specifies whether the dataset is for training, validation, or testing.
        - transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version.
        - pre_transform (callable, optional): A function/transform that is applied to the Data object before saving it to disk.
        - condition_type (str): Specifies the type of condition used for the graph. Can be 'graph', 'image', or 'image_resnet34'.
        """

        super(GraphDataset, self).__init__(transform, pre_transform)

        self.condition_type = condition_type
        self.data_type = data_type

        if condition_type == 'graph':
            self.folder_path = '/local_datasets/graph_condition_train_datasets/' + self.data_type
            self.folder_path = '/local_datasets/gt_train_datasets/' + self.data_type
        elif condition_type == 'image':
            self.folder_path = '/local_datasets/image_condition_train_datasets/' + self.data_type
        elif condition_type == 'image_resnet34':
            self.folder_path = '/local_datasets/gt_train_datasets/' + self.data_type
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
        self.data_length = len(self.gpickle_files)
        print(self.data_length)

    def get(self, idx):
        """
        Gets the graph data object at a specific index in the dataset.

        Parameters:
        - idx (int): The index of the data object to retrieve.

        Returns:
        - Data object or tuple: Depending on the data type, returns either a Data object for the graph data or a tuple containing the Data object, the path to polygon data, and the filename for test data.
        """

        if self.data_type == 'train' or self.data_type == 'val':
            load_path = self.folder_path + '/' + self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)

            graph = self.graph
            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)

            if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
                condition = torch.tensor(np.array(graph.graph['condition']), dtype=torch.float32)
                condition = condition.unsqueeze(0)
                condition = condition.repeat(3, 1, 1)
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

            data = Data(node_features=node_features,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

            return data
        else:
            load_path = self.folder_path + '/' + self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)
            graph = self.graph

            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)

            if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
                condition = torch.tensor(np.array(graph.graph['condition']), dtype=torch.float32)
                condition = condition.unsqueeze(0)
                condition = condition.repeat(3, 1, 1)
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

            data = Data(node_features=node_features,
                        building_mask=building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=graph.number_of_nodes())

            polygon_path = self.gpickle_files[idx].replace('.gpickle', '.pkl')
            return (data, polygon_path, self.gpickle_files[idx])

    def len(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of graph data objects in the dataset.
        """

        return self.data_length
