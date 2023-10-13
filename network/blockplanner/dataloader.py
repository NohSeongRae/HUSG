from torch_geometric.data import Data, Dataset
import numpy as np
import networkx as nx
import os

class BlockplannerDataset(Dataset):
    def __init__(self, n_building):
        super().__init__()

        self.dataset_path = './dataset/'
        self.file_names = [f for f in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path, f))]
        self.n_building = n_building

    def __getitem__(self, index):
        tmp_graph = nx.read_gpickle(os.path.join(self.dataset_path, '{}.gpickle'.format(index)))
        num_nodes = tmp_graph.number_of_nodes()

        node_index = np.stack((np.arange(num_nodes) / num_nodes, np.arange(num_nodes) / num_nodes), axis=1)

        adj_matrix = nx.adjacency_matrix(tmp_graph)

        x = list(nx.get_node_attributes(tmp_graph, 'x').items())
        y = list(nx.get_node_attributes(tmp_graph, 'y').items())
        w = list(nx.get_node_attributes(tmp_graph, 'w').items())
        h = list(nx.get_node_attributes(tmp_graph, 'h').items())
        n = list(nx.get_node_attributes(tmp_graph, 'n').items())
        geometry = np.stack((x, y, w, h, n), 1)

        semantic = list(nx.get_node_attributes(tmp_graph, 'semantic').items())
        semantic = np.zeros((len(semantic), self.n_building))

        aspect_ratio = np.array(list(nx.get_node_attributes(tmp_graph, 'aspect_ratio').items()))

        node_exists_prob = np.array(list(nx.get_node_attributes(tmp_graph, 'node_exists_prob').items()))

        return Data(geometry=geometry, semantic=semantic, adj_matrix=adj_matrix, node_index=node_index,
                    aspect_ratio=aspect_ratio, node_exists_prob=node_exists_prob)

    def __len__(self):
        return len(self.file_names)
