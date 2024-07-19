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
            self.folder_path = '/local_datasets/urban_datasets/datasets/ours_graph_datasets/' + self.data_type
        elif condition_type == 'image':
            self.folder_path = '/local_datasets/urban_datasets/datasets/ours_graph_datasets/' + self.data_type
        elif condition_type == 'image_resnet34':
            self.folder_path = '/local_datasets/urban_datasets/datasets/ours_graph_datasets/' + self.data_type

        if self.data_type == 'test':
            # self.folder_path = './large_scale_datasets/image_condition_train_datasets/'
            # self.folder_path = './synthetic_datasets_large_scale/'
            self.folder_path = './synthetic_datasets/'
            self.folder_path = './rebuttal_random_graph_generation_outputs/'
            # self.folder_path = '/local_datasets/urban_datasets/datasets/ours_graph_datasets/' + self.data_type
            # self.folder_path = './random_graph_generation_datasets'

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

            # building_n = torch.sum(building_masks).item()
            # boundary_n = building_masks.shape[0] - building_n
            # pooled_boundary_n = (boundary_n + 1) // 2
            #
            # pooled_node_features = torch.zeros((pooled_boundary_n + building_n, 5))
            # for i in range(pooled_boundary_n):
            #     for ii in range(2):
            #         if i * 2 + ii < boundary_n:
            #             pooled_node_features[i, 0] += node_features[i * 2 + ii, 0]
            #             pooled_node_features[i, 1] += node_features[i * 2 + ii, 1]
            #             pooled_node_features[i, 2] += node_features[i * 2 + ii, 2]
            #             pooled_node_features[i, 3] += node_features[i * 2 + ii, 3]
            #             if pooled_node_features[i, 4] == 0:
            #                 pooled_node_features[i, 4] += node_features[i * 2 + ii, 4]
            #         if ii == 1:
            #             pooled_node_features[i, 0] /= 2
            #             pooled_node_features[i, 1] /= 2
            #             pooled_node_features[i, 2] /= 2
            #             pooled_node_features[i, 3] /= 2
            # for i in range(building_n):
            #     pooled_node_features[i + pooled_boundary_n, 0] += node_features[i, 0]
            #     pooled_node_features[i + pooled_boundary_n, 1] += node_features[i, 1]
            #     pooled_node_features[i + pooled_boundary_n, 2] += node_features[i, 2]
            #     pooled_node_features[i + pooled_boundary_n, 3] += node_features[i, 3]
            #     pooled_node_features[i + pooled_boundary_n, 4] += node_features[i, 4]
            #
            # pooled_building_masks = torch.zeros((pooled_boundary_n + building_n, 1), dtype=torch.long)
            # pooled_building_masks[pooled_boundary_n:, 0] = 1
            #
            # if self.condition_type == 'image' or self.condition_type == 'image_resnet34':
            #     condition = torch.tensor(np.array(graph.graph['condition']), dtype=torch.float32)
            #     condition = condition.unsqueeze(0)
            #     condition = condition.repeat(3, 1, 1)
            # else:
            #     condition_graph = graph.graph['condition']
            #     condition_edge_index = nx.to_scipy_sparse_matrix(condition_graph).tocoo()
            #     condition_edge_index = torch.tensor(np.vstack((condition_edge_index.row, condition_edge_index.col)),
            #                                         dtype=torch.long)
            #     condition_street_feature = torch.tensor(
            #         np.array([condition_graph.nodes[node]['chunk_features'] for node in condition_graph.nodes()]),
            #         dtype=torch.float32)
            #
            #     condition = Data(condition_street_feature=condition_street_feature,
            #                      edge_index=condition_edge_index,
            #                      num_nodes=condition_graph.number_of_nodes())
            #
            # # edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
            # # edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)
            # adj_matrix = nx.to_numpy_array(graph)
            # boundary_adj_matrix = adj_matrix[:boundary_n, :boundary_n]
            # building_adj_matrix = adj_matrix[boundary_n:, boundary_n:]
            # bb_adj_matrix = adj_matrix[boundary_n:, :boundary_n]
            #
            # pooled_n_boundary = (boundary_n + 1) // 2
            # pooled_boundary_adj_matrix = np.zeros((pooled_n_boundary, pooled_n_boundary))
            # for i in range(pooled_n_boundary):
            #     for j in range(pooled_n_boundary):
            #         for ii in range(2):
            #             for jj in range(2):
            #                 if i * 2 + ii < boundary_n and j * 2 + jj < boundary_n:
            #                     pooled_boundary_adj_matrix[i, j] += boundary_adj_matrix[i * 2 + ii, j * 2 + jj]
            #                     if pooled_boundary_adj_matrix[i, j] > 1:
            #                         pooled_boundary_adj_matrix[i, j] = 1
            #
            # pooled_bb_adj_matrix = np.zeros((building_n, pooled_n_boundary))
            # for i in range(building_n):
            #     for j in range(pooled_n_boundary):
            #         for ii in range(2):
            #             if j * 2 + ii < boundary_n:
            #                 pooled_bb_adj_matrix[i, j] += bb_adj_matrix[i, j * 2 + ii]
            #                 if pooled_bb_adj_matrix[i, j] > 1:
            #                     pooled_bb_adj_matrix[i, j] = 1
            #
            # pooled_adj_matrix = np.zeros((pooled_n_boundary + building_n, pooled_n_boundary + building_n))
            # pooled_adj_matrix[:pooled_n_boundary, :pooled_n_boundary] = pooled_boundary_adj_matrix
            # pooled_adj_matrix[pooled_n_boundary:, pooled_n_boundary:] = building_adj_matrix
            # pooled_adj_matrix[pooled_n_boundary:, :pooled_n_boundary] = pooled_bb_adj_matrix
            # pooled_adj_matrix[:pooled_n_boundary, pooled_n_boundary:] = pooled_bb_adj_matrix.T
            #
            # import scipy.sparse as sp
            # sparse_matrix = sp.csr_matrix(pooled_adj_matrix)
            # coo_matrix = sparse_matrix.tocoo()
            # edge_index = torch.tensor(np.vstack((coo_matrix.row, coo_matrix.col)), dtype=torch.long)

            building_n = torch.sum(building_masks).item()
            boundary_n = building_masks.shape[0] - building_n
            unpooled_boundary_n = boundary_n * 2

            if unpooled_boundary_n > 200:
                x = 200
            else:
                x = unpooled_boundary_n

            unpooled_node_features = torch.zeros((x + building_n, 5))
            for i in range(x):
                unpooled_node_features[i, 0] += node_features[i // 2, 0]
                unpooled_node_features[i, 1] += node_features[i // 2, 1]
                unpooled_node_features[i, 2] += node_features[i // 2, 2]
                unpooled_node_features[i, 3] += node_features[i // 2, 3]
                unpooled_node_features[i, 4] += node_features[i // 2, 4]

            for i in range(building_n):
                if unpooled_boundary_n > 200:
                    x = 200
                else:
                    x = unpooled_boundary_n
                unpooled_node_features[i + x, 0] += node_features[i, 0]
                unpooled_node_features[i + x, 1] += node_features[i, 1]
                unpooled_node_features[i + x, 2] += node_features[i, 2]
                unpooled_node_features[i + x, 3] += node_features[i, 3]
                unpooled_node_features[i + x, 4] += node_features[i, 4]

            if unpooled_boundary_n > 200:
                x = 200
            else:
                x = unpooled_boundary_n
            unpooled_building_masks = torch.zeros((x + building_n, 1), dtype=torch.long)
            unpooled_building_masks[x:, 0] = 1

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

            # edge_index = nx.to_scipy_sparse_matrix(graph).tocoo()
            # edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)
            adj_matrix = nx.to_numpy_array(graph)
            boundary_adj_matrix = adj_matrix[:boundary_n, :boundary_n]
            building_adj_matrix = adj_matrix[boundary_n:, boundary_n:]
            bb_adj_matrix = adj_matrix[boundary_n:, :boundary_n]

            unpooled_n_boundary = (boundary_n + 1) // 2
            unpooled_boundary_adj_matrix = np.zeros((x, x))
            for i in range(x):
                for j in range(x):
                    unpooled_boundary_adj_matrix[i, j] = boundary_adj_matrix[i // 2, j // 2]

            unpooled_bb_adj_matrix = np.zeros((building_n, x))
            for i in range(building_n):
                for j in range(x):
                    unpooled_bb_adj_matrix[i, j] = bb_adj_matrix[i, j // 2]

            if unpooled_n_boundary > 200:
                unpooled_boundary_adj_matrix = unpooled_boundary_adj_matrix[:200, :200]
                unpooled_bb_adj_matrix = unpooled_bb_adj_matrix[:, :200]

            unpooled_adj_matrix = np.zeros((x + building_n, x + building_n))
            unpooled_adj_matrix[:x, :x] = unpooled_boundary_adj_matrix
            unpooled_adj_matrix[x:, x:] = building_adj_matrix
            unpooled_adj_matrix[x:, :x] = unpooled_bb_adj_matrix
            unpooled_adj_matrix[:x, x:] = unpooled_bb_adj_matrix.T

            import scipy.sparse as sp
            sparse_matrix = sp.csr_matrix(unpooled_adj_matrix)
            coo_matrix = sparse_matrix.tocoo()
            edge_index = torch.tensor(np.vstack((coo_matrix.row, coo_matrix.col)), dtype=torch.long)

            data = Data(node_features=unpooled_node_features,
                        building_mask=unpooled_building_masks, condition=condition,
                        edge_index=edge_index, num_nodes=x + building_n)

            return data
        else:
            load_path = self.folder_path + '/' + self.gpickle_files[idx]
            with open(load_path, 'rb') as f:
                self.graph = pickle.load(f)
            graph = self.graph
            print(len(graph.nodes))
            node_features = torch.tensor(np.array([graph.nodes[node]['node_features'] for node in graph.nodes()]),
                                         dtype=torch.float32)
            building_masks = torch.tensor(np.array([graph.nodes[node]['building_masks'] for node in graph.nodes()]),
                                          dtype=torch.long)

            print(node_features[0])
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